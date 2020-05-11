// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Transforms a given function into CSSA form, optionally coalescing copies
// where possible.

#define DEBUG_TYPE "do-cssa"
#define PASS_NAME "Insert phi elim copies"

#include "nvvm/Transforms/CSSA.h"
#include "nvvm/Analysis/MergeSets.h"
#include "nvvm/InitializePasses.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <forward_list>

using namespace llvm;

// TODO: Hook this up to some part of ClientAPI, probably NVVMCodeGenOptions.
static cl::opt<bool> CoalesceCopies("cssa-coalesce", cl::init(true),
                                    cl::Hidden);

static IntrinsicInst &createIntrinsic(Intrinsic::ID IID, ArrayRef<Type *> Tys,
                                      ArrayRef<Value *> Vals,
                                      IRBuilder<> &IRB) {
#if NVVM_VERSION < 700
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  Value *Intrin = Intrinsic::getDeclaration(M, IID, Tys);
  CallInst *CI = CallInst::Create(Intrin, Vals, "");
  IRB.GetInsertBlock()->getInstList().insert(IRB.GetInsertPoint(), CI);
  IRB.SetInstDebugLocation(CI);
  return *cast<IntrinsicInst>(CI);
#else
  return *cast<IntrinsicInst>(IRB.CreateIntrinsic(IID, Tys, Vals));
#endif
}

static IntrinsicInst &insertCopyImpl(ArrayRef<Type *> Tys,
                                     ArrayRef<Value *> Vals, IRBuilder<> &IRB) {
  return createIntrinsic(Intrinsic::nvvm_internal_copy, Tys, Vals, IRB);
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
    case Intrinsic::nvvm_internal_copy:
      return I->getOperand(1);
    case Intrinsic::ssa_copy:
      return I->getOperand(0);
    }
  }

  void coalesce() {
    LLVM_DEBUG(dbgs() << "Coalescing " << *I << "\n");
    I->replaceAllUsesWith(getSource());
    // Delete useless parallel copy markers.
    if (I->getIntrinsicID() == Intrinsic::nvvm_internal_copy &&
        I->getOperand(0)->hasOneUse()) {
      auto *Marker = cast<IntrinsicInst>(I->getOperand(0));
      I->eraseFromParent();
      Marker->eraseFromParent();
    } else
      I->eraseFromParent();
  }
};

struct CongValue {
  Instruction *I;
  unsigned DFSIn;
  unsigned DFSOut;
  unsigned LocalNum;

  bool dom(const CongValue &Other) const {
    if (DFSIn == Other.DFSIn)
      return LocalNum <= Other.LocalNum;
    return DFSIn <= Other.DFSIn && Other.DFSOut <= DFSOut;
  }

  // Returns true if this comes before Other in DPO. This is used primarily to
  // maintain cong class members in sorted order.
  bool dpoBefore(const CongValue &Other) const {
    return DFSIn < Other.DFSIn || LocalNum < Other.LocalNum;
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
#if NVVM_VERSION < 700
    O << *CV.I;
#else
    CV.I->printAsOperand(O, /* PrintType = */ false);
#endif
  }
  return O;
}

struct CSSA {
  Function &F;
  const DominatorTree &DT;
  const MergeSets &MS;
  // Un-coalesced copies.
  std::vector<PhiCopy> Copies;
  std::vector<PhiCopy> Coalescable;

  std::forward_list<CongClass> Classes;
  DenseMap<const Instruction *, CongClass *> ToClass;
  DenseMap<const Instruction *, unsigned> LocalNum;
  unsigned LocalCount = 0;

  CSSA(Function &F, const MergeSets &MS) : F(F), DT(MS.getDomTree()), MS(MS) {}

  CongValue fromInst(Instruction *I) {
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      if (II->getIntrinsicID() == Intrinsic::nvvm_internal_copy) {
        // For parallel copies, DFS and local numbering should come from the
        // parallel marker.
        auto *Marker = cast<IntrinsicInst>(I->getOperand(0));
        const DomTreeNode &N = *DT.getNode(Marker->getParent());
        return {I, N.getDFSNumIn(), N.getDFSNumOut(),
                LocalNum.find(Marker)->second};
      }
    const DomTreeNode &N = *DT.getNode(I->getParent());
    return {I, N.getDFSNumIn(), N.getDFSNumOut(), LocalNum.find(I)->second};
  }

  // Is Def live at N? TODO: If N is unreachable, this could be an issue.
  bool isLiveIn(const CongValue &N, const CongValue &Def) const {
    LLVM_DEBUG(dbgs() << "isLiveIn " << *N.I << ", " << *Def.I << "\n");
    assert(&N != &Def && Def.dom(N) &&
           "Should be part of the Budimlic dom-forest check.");
    // int_nvvm_internal_copy copies are defined at their parallel point.
    auto forwardToParallel = [](Instruction *I) {
      if (auto *II = dyn_cast<IntrinsicInst>(I))
        if (II->getIntrinsicID() == Intrinsic::nvvm_internal_copy)
          return cast<Instruction>(I->getOperand(0));
      return I;
    };

    // TODO: Liveness queries with parallel copies needs to be fixed.
    Instruction *NI = forwardToParallel(N.I), *DI = forwardToParallel(Def.I);
    LLVM_DEBUG(dbgs() << "isLiveIn " << *NI << ", " << *DI << "\n");

    // Edge cases: 1) nb == db; 2) nb == ub where u is a use of def.

    if (NI->getParent() == DI->getParent())
      return std::any_of(DI->user_begin(), DI->user_end(), [&](const User *U) {
        // If U is in the same block, check DPONum for local dominance.
        // Otherwise, NI must dom U since NI post-dom DI (easy proof by
        // contradiction).
        auto *I = cast<Instruction>(U);
        return I->getParent() != NI->getParent() ||
               N.LocalNum < LocalNum.find(I)->second;
      });

    const DomTreeNode *DB = DT.getNode(DI->getParent());
    const MergeSets::MergeInfo &M = MS.getMergeInfo(*NI->getParent());
    BitVector Mr = M.second;
    Mr.set(M.first);

    for (auto U = DI->use_begin(); U != DI->use_end(); ++U) {
      const BasicBlock *BB;
      // Figure out the basic block containing this use.
      if (const auto *PN = dyn_cast<PHINode>(U.getUse().getUser())) {
        BB = PN->getIncomingBlock(U.getUse());
        if (NI->getParent() == BB)
          // Phi uses are semantically at the end of BB.
          return true;
      } else if (const auto *I = dyn_cast<Instruction>(U.getUse().getUser())) {
        BB = I->getParent();
        // Account for nb == ub.
        if (NI->getParent() == BB) {
          if (DT.dominates(NI, I))
            return true;
          else
            // No local dominance, cannot be live at ni.
            continue;
        }
      } else
        llvm_unreachable("How can a non-inst use an inst?");

      assert(DT.dominates(DB, DT.getNode(BB)) &&
             "How can def DI not dominate use U?");

      for (const DomTreeNode *UB = DT.getNode(BB); UB != DB; UB = UB->getIDom())
        if (Mr.test(MS.getIndex(*UB->getBlock())))
          // Some block in merge(N) + {N} doms U.
          return true;
    }

    return false;
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
      if ((IA < A.size() && IB < B.size() && A[IA].dpoBefore(B[IB])) ||
          IB >= B.size())
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
      if ((IA < A.size() && IB < B.size() && A[IA].dpoBefore(B[IB])) ||
          IB >= B.size()) {
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

        // TODO: Enhancement: Optimize for singleton vreg classes, which could
        // be common.
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

    return true;
  }

  // Initially, we want to split a phi node into operand and value copies, and
  // group phi + copies into a single congruence class. TODO: This is done
  // separately from copy insertion to work around a circular dependency b/n
  // initial copy DPO num and dpoNum() needing a stable instruction list. The
  // whole initialization process should be refactored.
  CSSA &initPhiCongClass() {
    auto add = [&](IntrinsicInst &Copy, CongClass &PhiCls) {
      assert(Copy.getIntrinsicID() == Intrinsic::nvvm_internal_copy);
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
        PC.Members[0].LocalNum = LocalNum.find(&PN)->second;

        assert(PN.hasOneUse() && "Should only be used by a copy.");
        Copies.push_back({cast<IntrinsicInst>(*PN.user_begin())});
        // add(*cast<IntrinsicInst>(*PN.user_begin()), PC);
        for (unsigned PredNum = 0; PredNum < PN.getNumIncomingValues();
             PredNum += 1)
          add(*cast<IntrinsicInst>(PN.getIncomingValue(PredNum)), PC);

        std::sort(PC.Members.begin(), PC.Members.end(),
                  [](const CongValue &A, const CongValue &B) {
                    return A.dpoBefore(B);
                  });
        LLVM_DEBUG(dbgs() << "Sorted class:" << PC << "\n");
      }
    }
    return *this;
  }

  void insertPhiCopies(BasicBlock &BB) {
    if (BB.empty() || !isa<PHINode>(BB.begin()))
      return;
    const DomTreeNode &N = *DT.getNode(&BB);

    // Our goal is to break phi live ranges by inserting a copy for each
    // (unique) operand and a final copy for the phi itself. Unfortunately, phis
    // of a given block are not currently required to have the same operand
    // orderings.

    IRBuilder<> NonPhi(BB.getFirstNonPHI());

    for (BasicBlock *Pred : predecessors(&BB)) {
      IRBuilder<> IRB(Pred->getTerminator());
      CallInst *Parallel = &createIntrinsic(
          Intrinsic::nvvm_internal_parallel_copy, None, None, IRB);

      // Insert operand copies.
      for (Instruction &I : BB) {
        if (!isa<PHINode>(I))
          break;
        auto &PN = cast<PHINode>(I);
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

  // TODO: This should be idempotent.
  CSSA &insertCopies() {
#if NVVM_VERSION < 700
    DT.DT->updateDFSNumbers();
#else
    DT.updateDFSNumbers();
#endif
    for (BasicBlock &BB : F)
      insertPhiCopies(BB);
    LLVM_DEBUG(dbgs() << "CSSA " << F.getName() << "\n" << F << "\n");
    verifyFunction(F);
    return *this;
  }

  unsigned nextLocalNum(const Instruction &I) {
    unsigned Ret = LocalCount;
    // Leave a gap for first non-phi pcopy marker.
    LocalCount += 2;
    LocalNum.insert({&I, Ret});
    return Ret;
  }

  CSSA &localNum() {
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        nextLocalNum(I);
    return *this;
  }
};

struct CSSALegacyPass : public FunctionPass {
  static char ID;

  CSSALegacyPass() : FunctionPass(ID) {
    initializeCSSALegacyPassPass(*PassRegistry::getPassRegistry());
  }

#if NVVM_VERSION < 700
  const char *getPassName() const override { return PASS_NAME; }
#else
  StringRef getPassName() const override { return PASS_NAME; }
#endif

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MergeSetsPass>();
  }

  bool runOnFunction(Function &F) override {
#if NVVM_VERSION >= 700
    if (skipFunction(F))
      return false;
#endif
    return CSSA(F, getAnalysis<MergeSetsPass>().getMS())
        .insertCopies()
        .localNum()
        .initPhiCongClass()
        .coalesceAllCopies();
  }
};

char CSSALegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsPass)
INITIALIZE_PASS_END(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)

Pass *nvvm::createCSSAPass() { return new CSSALegacyPass(); }
