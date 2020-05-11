// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Transforms a given function into CSSA form, optionally coalescing copies
// where possible.
//
// Issues with this:
//
// - Not simple:
//   - Maintaining mappings between cc and inst in both directions.
//   - Maintaining additional list of copies to check or to coalesce.
//   - Parallel copies for phi ops but llvm.ssa.copy for phis themselves.
// - Not extensible:
//   - Assumptions of phi copies only in some places?
// - Not efficient, redundant work:
//   - Inserting many copies then removing most.
//   - Creating singleton cc that could get merged soon after.
//
// Reasoning for virtualization:
//
// Handle each phi one by one.
//
// Pretend all copies were already inserted, which mean that all x0-xn are
// initially *not* part of the cong class. Then one by one, attempt to coalesce
// each xi into the class. Note that it is guaranteed that at least one of them
// can work.

/*
x1 =

x2 = phi(x1, x3)
x3 = x2 + 1

use(x2)
use(x1)


x1 =
    x1' = x1

x2 = phi(x1, x3)
    x2 = x2'
x3 = x2 + 1
    x3' = x3

use(x2)
use(x1)


x2' = phi(x1, x3)
x3 = phi(...)
    x2 = x2'
    x3' = x3

use(x2)
use(x1)

a1 =
b1 =

a2 = phi(a1, v)
b2 = phi(b1, u)
x, y = a2, b2
u, v = x, y


a1 =
b1 =

a2 = phi(a1, v)
b2 = phi(b1, x)
x, y = a2, b2       2, 3 = 1, 2
v = y               1 = 3

a = phi(a1, y)
b = phi(b1, z)
c = phi(c1, x)
x, y, z = a, b, c   3, 1, 2 = 1, 2, 3

a = phi(a1, y)
b = phi(b1, z)
c = phi(c1, x)
x, rr = a, c        3, 4 = 1, 3
y, z = b, rr        1, 2 = 2, 4         4=3, 3=1, 1=2, 2=4

a = phi(a1, y)
b = phi(b1, z)
c = phi(c1, ss)
x = a               4 = 1
y, z, ss = b, c, x  1, 2, 3 = 2, 3, 4   4=1, 1=2, 2=3, 3=4


*/

#define DEBUG_TYPE "do-cssa"
#define PASS_NAME "Insert phi elim copies"

#include "llvm/Transforms/Scalar/CSSA.h"
#include "llvm/Analysis/MergeSets.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <forward_list>

using namespace llvm;

// TODO: Hook this up to some part of ClientAPI, probably NVVMCodeGenOptions.
static cl::opt<bool> CoalesceCopies("cssa-coalesce", cl::init(true),
                                    cl::Hidden);

template <typename T>
static IntrinsicInst &createIntrinsic(Intrinsic::ID IID, ArrayRef<Type *> Tys,
                                      ArrayRef<Value *> Vals, T &&IRB) {
  return *cast<IntrinsicInst>(IRB.CreateIntrinsic(IID, Tys, Vals));
}

static IntrinsicInst &appendIntrin(Intrinsic::ID IID, ArrayRef<Type *> Tys,
                                   ArrayRef<Value *> Vals, BasicBlock &BB) {
  return createIntrinsic(IID, Tys, Vals, IRBuilder<>(&BB));
}

static IntrinsicInst &insertCopyImpl(ArrayRef<Type *> Tys,
                                     ArrayRef<Value *> Vals, IRBuilder<> &IRB) {
  return createIntrinsic(Intrinsic::internal_copy, Tys, Vals, IRB);
}

template <typename T>
static IntrinsicInst &insertCopy(PHINode &P, unsigned OpNum, Value *PCopy,
                                 T &&IRB) {
  auto &Copy =
      insertCopyImpl({P.getType()}, {PCopy, P.getIncomingValue(OpNum)}, IRB);
  P.setIncomingValue(OpNum, &Copy);
  return Copy;
}

struct PhiCopy {
  IntrinsicInst *I;

  Value *getSource() const {
    switch (I->getIntrinsicID()) {
    default:
      llvm_unreachable("Not a phi copy intrinsic.");
      break;
    case Intrinsic::internal_copy:
      return I->getOperand(1);
    case Intrinsic::ssa_copy:
      return I->getOperand(0);
    }
  }

  void coalesce() {
    LLVM_DEBUG(dbgs() << "Coalescing " << *I << "\n");
    I->replaceAllUsesWith(getSource());
    // Delete useless parallel copy markers.
    if (getMarker() && I->getOperand(0)->hasOneUse()) {
      I->eraseFromParent();
      getMarker()->eraseFromParent();
    } else
      I->eraseFromParent();
  }

  IntrinsicInst *getMarker() const {
    return I->getIntrinsicID() == Intrinsic::internal_copy ? I->getOperand(0)
                                                           : nullptr;
  }
};

struct LocalNumbering {
  DenseMap<const Instruction *, unsigned> ToNum;
  unsigned LocalNumNext = 0;

  LocalNumbering(Function &F) {
    // Leaves an empty numbering slot for phi value copies.
    for (const BasicBlock &BB : F) {
      auto It = BB.begin();
      for (; It != BB.end() && isa<PHINode>(It); ++It)
        addInst(It);
      LocalNumNext += 1;
      for (; It != BB.end(); ++It)
        addInst(It);
    }
  }

  void addInst(const Instruction &I) { ToNum.insert({&I, LocalNumNext++}); }

  unsigned getNum(const Instruction &I) const { return ToNum.find(&I)->second; }
};

// Newtype around an instruction that belongs to some interference-free class.
struct CongValue {
  Instruction *I;
  unsigned DFSIn;
  unsigned DFSOut;
  unsigned LocalNum;

  // TODO: int_internal_copy copies are defined at their parallel point.

  bool dom(const CongValue &Other) const {
    if (DFSIn == Other.DFSIn)
      return LocalNum <= Other.LocalNum;
    return DFSIn <= Other.DFSIn && Other.DFSOut <= DFSOut;
  }

  // Returns true if this comes before Other in DPO.
  bool dpoBefore(const CongValue &Other) const {
    return DFSIn < Other.DFSIn || LocalNum < Other.LocalNum;
  }

  bool operator!=(const CongValue &Other) const { return I != Other.I; }

  // Is Def live at N? TODO: If N is unreachable, this could be an issue.
  bool isLiveAt(const CongValue &N, const MergeSets &MS) const {
    LLVM_DEBUG(dbgs() << "Checking if " << *I << " is live at " << *N.I
                      << "\n");
    assert(*this != N && dom(N) &&
           "Should be part of the Budimlic dom-forest check.");
    // int_internal_copy copies are defined at their parallel point.
    auto forwardToParallel = [](Instruction *I) {
      if (auto *II = dyn_cast<IntrinsicInst>(I))
        if (II->getIntrinsicID() == Intrinsic::internal_copy)
          return cast<Instruction>(I->getOperand(0));
      return I;
    };

    // TODO: Liveness queries with parallel copies needs to be fixed.
    Instruction *NI = forwardToParallel(N.I), *DI = forwardToParallel(Def.I);
    LLVM_DEBUG(dbgs() << "isLiveIn " << *NI << ", " << *DI << "\n");

    // Edge cases: 1) nb == db; 2) nb == ub where u is a use of def.

    if (getParent() == N.getParent())
      // Shortcut: When N is the same block as us, N will dom any use that does
      // not live between us and N (easy proof by contradiction).
      return any_of(uses(), [&](const Use &U) {
        auto *I = cast<Instruction>(U.getUser());
        return I->getParent() != N.getParent() ||
               N.DPONum < DPONum.find(I)->second;
      });

    // !M means N is unreachable, in which case the answer would still be false.
    if (const MergeInfo *M = N.getMergeInfo(MS)) {
      BitVector Mr = M->selfUnion();
      return any_of(uses(), [&](const Use &U) {
        const BasicBlock &BB = getUseBB(U);
        if (getParent() == &BB)
          // Local dominance check required. Phi uses exist at end of block.
          return isa<PHINode>(U.getUser()) ||
                 DPONum < DPONum.find(U.getUser())->second;
        for (const DomTreeNode *UB = MS.getDomTree().getNode(&BB);
             UB->getBlock() != DI->getParent(); UB = UB->getIDom())
          if (Mr.test(MS.getMergeInfo(*UB).BitPos))
            // Some block in merge(N) + {N} doms U.
            return true;
        return false;
      });
    }

    return false;
  }
};

struct CongClass {
  // This should always be sorted by DPO.
  std::vector<CongValue> Members;

  unsigned size() const { return Members.size(); }

  CongValue &operator[](unsigned Idx) { return Members[Idx]; }

  const CongValue &operator[](unsigned Idx) const { return Members[Idx]; }
};

raw_ostream &operator<<(raw_ostream &O, const CongClass &CC) {
  for (const CongValue &CV : CC.Members) {
    O << " ";
    CV.I->printAsOperand(O, /* PrintType = */ false);
  }
  return O;
}

// Pieces

struct CongValue, CongClass;

struct Coalescer {
  std::forward_list<CongClass> Classes;
  DenseMap<const Instruction *, CongClass *> ToClass;

  bool canCoalesce(const CongClass &, const CongClass &) const;

  bool tryCoalesce(const CongClass &, const CongClass &);
};

struct VirtCoalescer {
  std::forward_list<CongClass> Classes;

  void insertCSSA(Function &F) {
    // Insert all value copies. We need to do this to ensure that virtualized
    // phi vreg classes are fully isolated. If these are not fully isolated,
    // then it is possible that a) false interferences and/or b) incorrect cong
    // class members upon merge.
    for (BasicBlock &BB : F) {
      for (PHINode &PN : BB.phis()) {
        // Create at insert point.
        IntrinsicInst *Copy = createCopy(PN);
        PN.replaceAllUsesWith(Copy);
        Copy->setOperand(0, &PN);
      }
    }

    // want to be able to traverse like this, to avoid needing to traverse
    // bbs/phis or do more val-to-cc densemap lookups than we need.
    //
    // in agg coalescing, we can iterate over copies in any order we want, and
    // only once.  proof is to think about coalescing on an igraph. we only
    // coalesce at affinity edges, and each coalescing grows some iregion a bit
    // larger.  suppose by contradiction. then some affinity edge q could only
    // be coalescsed *after* another edge p. but that does not make sense.
    // result: we can choose to iterate over copies in phi-order so that we
    // don't have to keep reloading copy source's cong class.
    //
    // in that order in mat-coal, the initial phi class will contain a fully
    // isolated phi, which really means that we will never a merge a phi class
    // with something else until our iteration reaches that phi first. this is
    // equivalent to boissinot's "one should never have to test with..." stmt
    // except we do it right via the initial full value copy insertion. result:
    // we don't even have to pre-allocate a cc for a phi until iteration reaches
    // it.
    for (BasicBlock &BB : F)
      for (PHINode &PN : BB.phis())
        coalesceVirtual(PN);
  }

  // RAUW a copy from IR and optionally pop from its vreg class. Popping mainly
  // needed by incremental coalescing.
  void rauw(IntrinsicInst &Copy, CongClass *CC = nullptr) {
    LLVM_DEBUG(dbgs() << "RAUW-ing " << Copy << "\n");
    if (CC) {
      LLVM_DEBUG(dbgs() << "Popping from vreg class.\n");
      // In-place coalescing: Remove Copy from vreg class and IR.
      auto It =
          find_if(*CC, [](const CongValue &Val) { return Val.I == Copy; });
      assert(It != CC->end());
      CC->erase(It);
    }

    Copy.replaceAllUsesWith(Copy.getOperand(1));
    auto *Marker = cast<Instruction>(Copy.getOperand(0));
    Copy.eraseFromParent();

    // Delete useless parallel copy markers.
    if (Marker->use_empty())
      Marker->eraseFromParent();
  }

  void coalesceVirtual(PHINode &PN) {
    // Init PN's vreg class.
    CongClass &CC = Class.emplace_back();
    for (unsigned OpNum = 0; OpNum < PN.getNumIncomingValues(); OpNum += 1) {
      IntrinsicInst *Copy = createCopy(PN, OpNum);
      CC.add(Copy);
    }
    CC.add(PN);
    std::sort(CC.begin(), CC.end(), [](const CongValue &A, const CongValue &B) {
      return A.dpoBefore(B);
    });

    // Operand copy coalescing.
    for (Use &U : PN.operands()) {
      auto *Copy = cast<IntrinsicInst>(U);
      CongClass &SrcCC = getCongClass(Copy->getOperand(1));
      if (&SrcCC == &CC) {
        // Possible if SrcCC and CC had multiple affinities, e.g. if a phi had
        // the same incoming value for more than one pred.
        rauw(*Copy, &CC);
      } else if (!intersects(CC, SrcCC)) {
        rauw(*Copy, &CC);
        mergeInto(CC, SrcCC);
      }
    }

    // Value copy coalescing.
    assert(PN.hasOneUse());
    auto *ValCopy = cast<IntrinsicInst>(PN.user_begin());
    CongClass &DstCC = getCongClass(ValCopy);
    if (&DstCC == &CC) {
      // TODO: Not sure if this is ever taken.
      rauw(*ValCopy, &DstCC);
    } else if (!intersects(CC, DstCC)) {
      rauw(*ValCopy, &DstCC);
      mergeInto(CC, DstCC);
    }

    // At this point: CC has been updated; appropriate copies removed from CC
    // and IR;

    // Goals:
    //
    // - if a copy has been coalesced, we don't want to keep it around in the
    // dfs vec. this would just slow down interference checks and extra
    // memory.
    // - want to do virtualized.
    //
    // try mat-coal: build dfsvec of inserted copies, then check each operand
    // copy's source class, try to merge, if merged then reach in and remove
    // that copy from dfsvec and also erase inserted copy.
    //
    // problems: removal costs another o(n) scan plus memmove/cpy.
    for (unsigned OpNum = 0; OpNum < PN.getNumIncomingValues(); OpNum += 1) {
      // flip operand use off, merge operand class with ours, also kill off
      // this vcopy.
    }

    for (PHINode &PN : BB.phis()) {
      // In a non-virtualized setting, PN's vreg class would initially contain
      // copies + isolated phi and these are guaranteed not to intersect. Then
      // one-by-one we check if we can coalesce each copy by merging copy
      // source into class and checking for intersection. Only source of
      // interference is op-op or op-copy.
      //
      // Would virtualization miss any op-copy interferences? Is op-copy
      // possible?
      //
      // Unknown. Not investing more time towards proving this.
      //
      // Initializing with virtual copies and having to handle materialization
      // seems only marginally less expensive and more complex than
      // materialized coalescing (inserting real copies then coalescing). So
      // let's go with the latter once more.
      //
      // Choices for mat-coal:
      //
      // - insert phi-by-phi.
      // - insert for all phis, block-by-block.
      //
      // Looks like we will have to insert all value copies because when
      // virt-coal we could be testing against another phi cong class that has
      // not fully materialized yet.
      //
      // ;w
      //

      CongClass &PCC = initializePhiClass(PN);
      // PCC: {c1, c2, ..., p0}
      // materialized coalesce: c1 = copy(a1), try {a1, c1, ...}
      // virt coal: try {a1, c2, ...}. if it works,
      // so focus on the interference check for operands.
      // requires a ccdfsvec,
      //
      // type CC = Vec Value
      // type AllCC = DenseMap Value CC
      //
      // self_interf :: CC -> Bool
      // interf1 :: CC -> Value -> Maybe CC
      // interf :: CC -> CC -> Maybe CC
      //
      // get_class :：Value -> CC
      //
      // try_coal :: AllCC -> Copy -> AllCC
      // try_coal ccs (Copy dst src) = case interf (get_class ccs dcc)
      // (get_class ccs scc) of
      //     None -> ccs
      //     Maybe new -> update (update ccs dst new) src new

      for (unsigned Idx = 0; Idx < PN.getNumIncomingValues(); Idx += 1) {
        coalesceOperandCopy(PCC, PN, Idx);
      }
      insertOperandCopy(PN, Idx);

      // Coalesce value copy.
      assert(PN.hasOneUse() && "Only use of phi should be its value copy.");
      auto *Copy = cast<IntrinsicInst>(PN.user_begin());
    }
  }

  void insertOperandCopy(PHINode &PN, unsigned Idx) {
    // pick out phi class
    // test if that op Idx interferes with the class.
    // if so, replace with copy.
    // if not, add into class.
  }

  void howThisWorks(PHINode &);
};

void VirtCoalescer::coalesceVirtual(PHINode &PN, CongClass &PCC) {
  // PCC is PN's coalescing class. Look for a non-interfering set of PN's
  // operands and add to PCC.
  for (unsigned OpNum = 0; OpNum < PN.getNumIncomingValues(); OpNum += 1) {
    Use &U = PN.getOperandUse(OpNum);
    Value *V = U.get();
    U.set(nullptr);

    // Check if virtual copy can be coalesced. Do it if so.
    if (!tryCoalesce(*V, PCC)) {
      // This operand interferes, insert a copy to break its live range.
      IntrinsicInst *Copy = PhiSetup.insertCopy(V);
      U.set(Copy);
    } else
      U.set(V);
  }
}

void VirtCoalescer::howThisWorks(PHINode &PN) {
  // Roughly:

  CongClass &PCC = *getCongClass(PN);

  // Currently only operands are virtual. Should update coalescing classes as
  // needed.
  coalesceVirtual(PN, PCC);

  // Check if value copy can be coalesced and update CC.
  auto *ValCopy = cast<IntrinsicInst>(PN.user_begin());
  tryCoalesce(*ValCopy, PCC);
}

static const BasicBlock &getUseBB(const Use &U) {
  assert(isa<Instruction>(U.get()) && "Only use on instruction uses.");
  if (const auto *PN = dyn_cast<PHINode>(U.getUser()))
    return PN->getIncomingBlock(*U);
  else if (const auto *I = dyn_cast<Instruction>(U.getUser()))
    return I->getParent();
  else
    llvm_unreachable("How can a non-inst use an inst?");
}

// We want to transform p = phi(op_0, ..., op_n) into:
//
//   x_0 = phi(x_1, x_{n+1}) where x_i = copy(op_i)
//   p = copy(x_0)
struct PhiSetup {
  // Basic block containing phis that we want to set up.
  BasicBlock &BB;
  // Position of first non-phi where phi-copy will be inserted.
  BasicBlock::iterator &NonPhi;
  CSSA &Opt;
  // Parallel copy markers for each predecessor of BB.
  SmallVector<std::pair<BasicBlock *, IntrinsicInst *>, 32> Markers;

  PhiSetup(BasicBlock &BB, CSSA &Opt)
      : BB(BB), NonPhi(BB.getFirstInsertionPt()), Opt(Opt) {
    for (BasicBlock *Pred : predecessors(BB))
      Markers.push_back({Pred, &appendIntrin(Intrinsic::internal_parallel_copy,
                                             None, None, *Pred)});
  }

  void breakPhiRanges() const { ; }

  // Insert copies; init PN's cong class.
  void setup(PHINode &PN) {
    CongClass &CC = CSSA.ToClass.insert(
        {&PN, addCongClass({&PN, N.getDFSNumIn(), N.getDFSNumOut()})});
    for (const auto &Pair : Markers) {
      unsigned OpNum = PN.getBasicBlockIndex(Pair.first);
      IntrinsicInst &Copy =
          appendIntrin(Intrinsic::internal_copy, {PN.getType()},
                       {Pair.second, PN.getIncomingValue(OpNum)}, *Pred);
    }
  }

  void insertPhiCopies(BasicBlock &BB) {
    if (BB.empty() || !isa<PHINode>(BB.begin()))
      return;
    const DomTreeNode &N = *DT.getNode(&BB);

    // Note that phis of a given block could have differing operand order.

    IRBuilder<> NonPhi(BB.getFirstNonPHI());

    for (BasicBlock *Pred : predecessors(&BB)) {
      IRBuilder<> IRB(Pred->getTerminator());
      CallInst *Parallel =
          &createIntrinsic(Intrinsic::internal_parallel_copy, None, None, IRB);

      // Insert operand copies.
      for (PHINode &PN : BB.phis()) {
        insertCopy(PN, PN.getBasicBlockIndex(Pred), Parallel, IRB);
        ToClass.insert(
            {&PN, addCongClass({&PN, N.getDFSNumIn(), N.getDFSNumOut()})});
      }
    }

    // Break phi live ranges.
    for (Instruction &I : BB) {
      if (!isa<PHINode>(I))
        break;
      auto &PN = cast<PHINode>(I);
      CallInst &P_ = createIntrinsic(Intrinsic::ssa_copy, {PN.getType()},
                                     {UndefValue::get(PN.getType())}, NonPhi);
      PN.replaceAllUsesWith(&P_);
      P_.setOperand(0, &PN);
    }
  }
};

// Main improvements with new approach:
//
// - operate one bb at a time, simplifies reasoning and reduces peak mem
// usage.
// - optimized cc-inst coalescing, no singleton ccs.
// - single intrinsic kind for all copies.
//
// Still inserting then removing most. Virtualization is more complex and
// might not be able to coalesce all copies, would at least need phi dest
// copies.
struct Coalescer {
  // Should only need exactly one CC per phi.
  std::forward_list<CongClass> Classes;
  DenseMap<const Instruction *, CongClass *> ToClass;

  // Simultaneously check for intersection between Dest and Src and attempt to
  // coalesce into Dest, returning true upon success. Exists to save an extra
  // iteration from a separate merge step.
  bool tryCoalesce(const CongClass &Dest, const CongClass &);

  // Optimized for a single instruction. TODO: Store singleton classes instead
  // of repeated look-ups.
  bool tryCoalesce(const CongClass &, CongValue &);

  void coalesceCopies(ArrayRef<Copy>);

  // Setup-related routines.
  void addCongClass();
};

struct CSSA {
  Function &F;
  const DominatorTree &DT;
  const MergeSets &MS;
  LocalNumbering LN;
  // Un-coalesced copies.
  std::vector<PhiCopy> Copies;
  std::vector<PhiCopy> Coalescable;

  std::forward_list<CongClass> Classes;
  DenseMap<const Instruction *, CongClass *> ToClass;

  CSSA(Function &F, const MergeSets &MS)
      : F(F), DT(MS.getDomTree()), MS(MS), LN(F) {}

  CongValue fromInst(Instruction *I) {
    const DomTreeNode &N = *DT.getNode(I->getParent());
    return {I, N.getDFSNumIn(), N.getDFSNumOut(), DPONum.find(I)->second};
  }

  Instruction &addInst(Instruction &I) {
    LocalNum.insert({&I, LocalNumNext++});
    return I;
  }

  // Given intersection-free sets A and B, decide if the combination of both
  // self-intersects.
  bool intersects(const CongClass &A, const CongClass &B) const {
    // TODO: This does Budimlic checks over A merged with B where both are
    // DPO-ordered, but we could also do the actual merge here to save an
    // extra iter. May not be worthwhile, though.

    LLVM_DEBUG(dbgs() << "Intersecting\n " << A << "\n  and" << B << "\n");

    // Second element of pair denotes origination from A or B.
    struct StackEntry {
      CongValue V;
      enum { FromA, FromB } Cls;
    };
    SmallVector<StackEntry, 32> Stack;

    for (unsigned IA = 0, IB = 0; IA < A.size() || IB < B.size();) {
      // Pick out the next element from either A or B in dom-forest order.
      //
      // inbounds(ia) && inbounds(ib) => {
      //   a[ia] < b[ib] => a[ia++],
      //   a[ia] >= b[ib] => b[ib++],
      // },
      // inbounds(ia) (but not ib) => a[ia++],
      // inbounds(ib) (but not ia) => b[ib++]
      StackEntry Cur;
      if ((IA < A.size() && IB < B.size() && A[IA] < B[IB]) || IB >= B.size())
        Cur = {A[IA++], StackEntry::FromA};
      else
        Cur = {B[IB++], StackEntry::FromB};
      LLVM_DEBUG(dbgs() << "Cur: " << *Cur.V.I << "\n");

      while (!Stack.empty() && !Stack.back().V.dom(Cur.V))
        Stack.pop_back();

      // Skip checking if TOS and Cur came from the same class since the fn
      // pre-condition already ensures no intersection.
      if (!Stack.empty() && Stack.back().Cls != Cur.Cls &&
          isLiveIn(Cur.V, Stack.back().V)) {
        LLVM_DEBUG(dbgs() << "Would intersect.\n");
        return true;
      }

      Stack.push_back(Cur);
    }
    return false;
  }

  void mergeInto(CongClass &A, CongClass &B) {
    // TODO: Optimize for the common case when one of A or B are singleton.
    CongClass Merged;
    unsigned IA = 0, IB = 0;
    for (; IA < A.size() || IB < B.size();) {
      if ((IA < A.size() && IB < B.size() && A[IA] < B[IB]) || IB >= B.size()) {
        Merged.Members.push_back(A[IA++]);
      } else {
        Merged.Members.push_back(B[IB++]);
        auto It = ToClass.find(Merged.Members.back().I);
        assert(It != ToClass.end() &&
               "Value-to-class out of sync from class-to-value.");
        It->second = &A;
      }
    }

    std::swap(Merged.Members, A.Members);
    B.Members.clear();
    LLVM_DEBUG(dbgs() << "Coalesced cong class: " << A << "\n");
  }

  CongClass *addCongClass(CongValue V) {
    Classes.push_front({{V}});
    return &Classes.front();
  }

  // Find I's congruence class or else create one and return it.
  CongClass &getCongClass(Instruction &I) {
    auto Pair = ToClass.insert({&I, nullptr});
    if (Pair.second)
      Pair.first->second = addCongClass(fromInst(&I));
    return *Pair.first->second;
  }

  bool coalesceAllCopies() {
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (unsigned Idx = 0; Idx < Copies.size(); Idx += 1) {
        LLVM_DEBUG(dbgs() << "Attempting to coalesce " << *Copies[Idx].I
                          << "\n");
        // TODO: Handle constants and arguments as though they were defined in
        // entry.
        if (!isa<Instruction>(Copies[Idx].getSource()))
          continue;

        // TODO: Optimize for the common case when one of CC or SrcCC are
        // singleton.
        CongClass &CC = getCongClass(*Copies[Idx].I);
        CongClass &SrcCC =
            getCongClass(*cast<Instruction>(Copies[Idx].getSource()));

        if (&CC == &SrcCC) {
          LLVM_DEBUG(dbgs() << "Trivially coalescable.\n");
          Coalescable.push_back(Copies[Idx]);
          std::swap(Copies.back(), Copies[Idx]);
          Copies.pop_back();
          Idx -= 1;

          Changed = true;
        } else if (!intersects(CC, SrcCC)) {
          LLVM_DEBUG(dbgs() << "No interference, can coalesce!\n");
          mergeInto(CC, SrcCC);

          Coalescable.push_back(Copies[Idx]);
          std::swap(Copies.back(), Copies[Idx]);
          Copies.pop_back();
          Idx -= 1;

          Changed = true;
        }
      }
    }

    if (Coalescable.empty())
      return false;

    // Coalesce!
    for (PhiCopy &PC : Coalescable)
      PC.coalesce();

    // TODO: Implement copy sequentialization in D2IR.
    checkSequentialization();
    return true;
  }

  // Initially, we want to split a phi node into operand and value copies,
  // and group phi + copies into a single congruence class. TODO: This is
  // done separately from copy insertion to work around a circular
  // dependency b/n initial copy DPO num and dpoNum() needing a stable
  // instruction list. The whole initialization process should be
  // refactored.
  CSSA &initPhiCongClass() {
    auto add = [&](IntrinsicInst &Copy, CongClass &PhiCls) {
      assert(Copy.getIntrinsicID() == Intrinsic::internal_copy);
      LLVM_DEBUG(dbgs() << "Init " << Copy << " to phi-class "
                        << *PhiCls.Members[0].I << "\n");
      Copies.push_back({&Copy});
      ToClass.insert({&Copy, &PhiCls});
      PhiCls.Members.push_back(fromInst(&Copy));
    };

    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end() && isa<PHINode>(It); ++It) {
        auto &PN = *cast<PHINode>(It);
        CongClass &PC = getCongClass(PN);
        assert(PC.size() == 1 && "Assumption: This is called immediately after "
                                 "insertPhiCopies and dpoNum.");
        PC.Members[0].DPONum = DPONum.find(&PN)->second;

        assert(PN.hasOneUse() && "Should only be used by a copy.");
        Copies.push_back({cast<IntrinsicInst>(*PN.user_begin())});
        // add(*cast<IntrinsicInst>(*PN.user_begin()), PC);
        for (unsigned PredNum = 0; PredNum < PN.getNumIncomingValues();
             PredNum += 1)
          add(*cast<IntrinsicInst>(PN.getIncomingValue(PredNum)), PC);

        std::sort(
            PC.Members.begin(), PC.Members.end(),
            // TODO: THIS IS TOTALLY WRONG! Dominance is not the correct sort
            // order.
            [](const CongValue &A, const CongValue &B) { return A.dom(B); });
        LLVM_DEBUG(dbgs() << "Sorted class:" << PC << "\n");
      }
    }
    return *this;
  }

  // TODO: This should be idempotent.
  void initializePhiCopies() {
    DT.updateDFSNumbers();
    for (BasicBlock &BB : F) {
      if (BB.empty() || !isa<PHINode>(BB.begin()))
        continue;
      PhiSetup PS(BB, *this);
      for (PHINode &PN : BB.phis())
        PS.setup(PN);
    }
  }

  bool run() {

    initializePhiCopies();

    LLVM_DEBUG(dbgs() << "After copy insertion:" << F << "\n");
    verifyFunction(F);

    coalesceAllCopies();
  }
};

struct CSSALegacyPass : public FunctionPass {
  static char ID;

  CSSALegacyPass() : FunctionPass(ID) {
    initializeCSSALegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MergeSetsWrapper>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    bool Changed = false;
    for (BasicBlock &BB : F) {
      CSSAify C(BB);
      for (PHINode &PN : BB.phis())
        Changed |= C.insertCopiesFor(PN);
    }
    return Changed;
  }
};

char CSSALegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsWrapper)
INITIALIZE_PASS_END(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)

Pass *nvvm::createCSSAPass() { return new CSSALegacyPass(); }
