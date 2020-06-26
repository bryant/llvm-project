// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Transforms a given function into CSSA form, optionally coalescing copies
// where possible.

#define DEBUG_TYPE "do-cssa"
#define PASS_NAME "Insert phi elim copies"
#define NVVM_VERSION 1100
#define nvvm_internal_copy internal_copy
#define nvvm_internal_parallel_copy internal_parallel_copy

#include "llvm/Transforms/Scalar/CSSA.h"
#include "llvm/Analysis/MergeSets.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#if NVVM_VERSION < 700
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Verifier.h"
#else
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#endif

#include <forward_list>

using namespace llvm;

// TODO: Hook this up to some part of ClientAPI, probably NVVMCodeGenOptions.
cl::opt<bool> CoalesceCopies("cssa-coalesce", cl::init(false), cl::Hidden);

static cl::opt<bool> DumpBefore("dump-before-cssa", cl::init(false),
                                cl::Hidden);

static IntrinsicInst &createIntrinsic(Intrinsic::ID IID, ArrayRef<Type *> Tys,
                                      ArrayRef<Value *> Vals,
                                      const Twine &Prefix, Instruction *Ins) {
  Module *M = Ins->getParent()->getParent()->getParent();
  Function *Intrin = Intrinsic::getDeclaration(M, IID, Tys);
  CallInst *CI = CallInst::Create(Intrin, Vals, Prefix, Ins);
  // TODO: Debug info updates.
  return *cast<IntrinsicInst>(CI);
}

struct PhiCopy {
  IntrinsicInst *I;

  PhiCopy(Instruction &I) : I(&cast<IntrinsicInst>(I)) {
    assert(this->I->getIntrinsicID() == Intrinsic::nvvm_internal_copy);
  }

  Value *getSource() const { return I->getOperand(1); }

  void setSource(Value &V) const { I->setOperand(1, &V); }
};

struct CongValue {
  Instruction *I;
  const DomTreeNode *N;
  unsigned LocalNum;

  bool dom(const CongValue &Other) const {
    if (N->getDFSNumIn() == Other.N->getDFSNumIn())
      return LocalNum <= Other.LocalNum;
    return N->getDFSNumIn() <= Other.N->getDFSNumIn() &&
           Other.N->getDFSNumOut() <= N->getDFSNumOut();
  }

  // Does this precede Other in DPO? Needed to sorted cong classes.
  bool dpoBefore(const CongValue &Other) const {
    if (N->getDFSNumIn() == Other.N->getDFSNumIn())
      return LocalNum < Other.LocalNum;
    return N->getDFSNumIn() < Other.N->getDFSNumIn();
  }

  BasicBlock &getParent() const { return *I->getParent(); }

  const DomTreeNode &getNode() const { return *N; }
};

struct CongClass {
  // This should always be sorted by DPO.
  std::vector<CongValue> Members;

  unsigned size() const { return Members.size(); }

  CongValue &operator[](unsigned Idx) { return Members[Idx]; }

  const CongValue &operator[](unsigned Idx) const { return Members[Idx]; }

  CongValue &add(CongValue CV) {
    Members.push_back(CV);
    return Members.back();
  }

  void erase(const Instruction &I) {
    auto It = std::find_if(Members.begin(), Members.end(),
                           [&](const CongValue &Val) { return Val.I == &I; });
    assert(It != Members.end());
    Members.erase(It);
  }

  void dump() const;
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

void CongClass::dump() const { dbgs() << *this; }

static BasicBlock *getUseBlock(Use &U) {
  if (auto *PN = dyn_cast<PHINode>(U.getUser()))
    return PN->getIncomingBlock(U);
  return cast<Instruction>(U.getUser())->getParent();
}

struct CSSA {
  Function &F;
  const DominatorTree &DT;
  const MergeSets &MS;

  std::forward_list<IntrinsicInst *> AllMarkers;
  std::forward_list<CongClass> Classes;

  // Each instruction belongs to a unique vreg class.
  DenseMap<const Instruction *, CongClass *> ToClass;

  DenseMap<const Instruction *, unsigned> LocalNum;
  unsigned LocalCount = 0;

  CSSA(Function &F, const MergeSets &MS) : F(F), DT(MS.getDomTree()), MS(MS) {}

  // TODO: Enhancement: Extend liveness checking to handle unreachables.
  const DomTreeNode *getDomNode(const BasicBlock *BB) const {
    const DomTreeNode *Ret = DT.getNode(BB);
    assert(Ret && "Found an edge case with unreachable blocks.");
    return Ret;
  }

  // Is Def live at N? TODO: Bug: Non-local dom checks will assert if N is
  // unreachable.
  bool isLiveIn(const CongValue &N, const CongValue &Def) const {
    LLVM_DEBUG(dbgs() << "isLiveIn " << *N.I << ", " << *Def.I << "\n");
    assert(&N != &Def && Def.dom(N) &&
           "Should be part of the Budimlic dom-forest check.");

    // Edge cases: 1) nb == db; 2) nb == ub where u is a use of def.

    if (&N.getParent() == &Def.getParent())
      // Shortcut: Check for any use of Def coming after N.
      return any_of(Def.I->users(), [&](const User *U) {
        auto *I = cast<Instruction>(U);
        return I->getParent() != &N.getParent() || isa<PHINode>(U) ||
               I->isTerminator() || N.LocalNum < LocalNum.find(I)->second;
      });

    BitVector Mr = MS.getMergeInfo(N.getNode()).selfUnion();
    for (auto Ut = Def.I->use_begin(); Ut != Def.I->use_end(); ++Ut) {
#if NVVM_VERSION < 700
      Use &U = Ut.getUse();
#else
      Use &U = *Ut;
#endif
      const BasicBlock *UB = getUseBlock(U);
      auto *I = cast<Instruction>(U.getUser());
      if (&N.getParent() == UB) {
        // Local dom checks. Note: This is the only time we have to worry about
        // pcopy nuances (since pcopy local num == pmarker local num)!
        if (isa<PHINode>(U.getUser()) || I->isTerminator() ||
            N.LocalNum < LocalNum.find(cast<Instruction>(I))->second)
          return true;
        // N !dom U, cannot be live from this use.
        continue;
      }

      assert(DT.dominates(&Def.getNode(), getDomNode(UB)) &&
             "How can Def not dominate use U?");
      for (const DomTreeNode *UN = getDomNode(UB); UN != &Def.getNode();
           UN = UN->getIDom())
        if (Mr.test(MS.getMergeInfo(*UN).BitPos))
          // Some block in merge(N) + {N} doms U.
          return true;
    }

    return false;
  }

  // Given intersection-free sets A and B, decide if the combination of both
  // self-intersects.
  bool intersects(const CongClass &A, const CongClass &B) const {
    LLVM_DEBUG(dbgs() << "Intersecting " << A << "\n        and " << B << "\n");
    // Second element is true if CongValue came from A otherwise false (from B).
    SmallVector<std::pair<CongValue, bool>, 32> Stack;

    for (unsigned IA = 0, IB = 0; IA < A.size() || IB < B.size();) {
      // Pick out the next element from either A or B in dom-forest order.
      //
      // inbounds(ia) && inbounds(ib) => {
      //   a[ia] < b[ib] => a[ia++],
      //   a[ia] >= b[ib] => b[ib++],
      // },
      // inbounds(ia) (but not ib) => a[ia++],
      // inbounds(ib) (but not ia) => b[ib++]
      std::pair<CongValue, bool> Cur;
      if ((IA < A.size() && IB < B.size() && A[IA].dpoBefore(B[IB])) ||
          IB >= B.size())
        Cur = {A[IA++], true};
      else
        Cur = {B[IB++], false};
      LLVM_DEBUG(dbgs() << "Cur: " << *Cur.first.I << "\n");

      while (!Stack.empty() && !Stack.back().first.dom(Cur.first))
        Stack.pop_back();

      // Skip checking if TOS and Cur came from the same class since the fn
      // pre-condition already ensures no intersection.
      if (!Stack.empty() && Stack.back().second != Cur.second &&
          isLiveIn(Cur.first, Stack.back().first)) {
        LLVM_DEBUG(dbgs() << "Would intersect.\n");
        return true;
      }

      Stack.push_back(Cur);
    }
    return false;
  }

  // Merge B into A. TODO: Enhancement: For singleton classes, this could be
  // logarithmic instead of linear. Otherwise, iter order is same as interf.
  void mergeInto(CongClass &A, CongClass &B) {
    LLVM_DEBUG(dbgs() << "mergeInto " << A << "\n  from " << B << "\n");
    if (!A.size() || !B.size())
      return;

    CongClass Merged;
    for (unsigned IA = 0, IB = 0; IA < A.size() || IB < B.size();) {
      if ((IA < A.size() && IB < B.size() && A[IA].dpoBefore(B[IB])) ||
          IB >= B.size()) {
        Merged.add(A[IA++]);
      } else {
        Merged.add(B[IB++]);
        auto It = ToClass.find(Merged.Members.back().I);
        assert(It != ToClass.end() &&
               "Value-to-class out of sync from class-to-value.");
        It->second = &A;
      }
    }

    std::swap(Merged.Members, A.Members);
    B.Members.clear();
    LLVM_DEBUG(dbgs() << "Coalesced cong class: " << A << "\n");
    for (const CongValue &CV : A.Members) {
      LLVM_DEBUG(dbgs() << "Val: " << *CV.I << " => ");
      auto It = ToClass.find(CV.I);
      assert(It != ToClass.end());
      assert(It->second->size());
      LLVM_DEBUG(dbgs() << *It->second << "\n");
    }
  }

  // Find I's congruence class or else create one and return it. Specify CV to
  // save DT and LocalNum lookups.
  CongClass &getCongClass(Instruction &I, Optional<CongValue> CV) {
    auto Pair = ToClass.insert({&I, nullptr});
    if (Pair.second) {
      Classes.push_front({{CV ? *CV
                              : CongValue{&I, getDomNode(I.getParent()),
                                          LocalNum.find(&I)->second}}});
      Pair.first->second = &Classes.front();
      LLVM_DEBUG(dbgs() << "Created new class for " << I << ":"
                        << Classes.front() << "\n");
    } else
      LLVM_DEBUG(dbgs() << "Got class for " << I << ":" << *Pair.first->second
                        << "\n");
    return *Pair.first->second;
  }

  unsigned localNumber(const Instruction &I) {
    return LocalNum.insert({&I, LocalCount++}).first->second;
  }

  // Insert and number a parallel copy marker.
  CongValue createMarker(BasicBlock &BB, Optional<Instruction *> Ins) {
    IntrinsicInst &Marker =
        createIntrinsic(Intrinsic::nvvm_internal_parallel_copy, {}, {}, "pmk",
                        Ins ? *Ins : BB.getTerminator());
    AllMarkers.push_front(&Marker);
    return {&Marker, getDomNode(&BB), localNumber(Marker)};
  }

  // Side concerns: Updating cong class with this inst; setting value op.
  CongValue insertCopy(Type &Ty, const CongValue &Marker,
                       Optional<Instruction *> Ins) {
    CongValue Ret = Marker;
    Ret.I = &createIntrinsic(Intrinsic::nvvm_internal_copy, {&Ty},
                             {Marker.I, UndefValue::get(&Ty)}, "pcp",
                             Ins ? *Ins : Marker.getParent().getTerminator());
    auto Pair = LocalNum.insert({Ret.I, Ret.LocalNum});
    assert(Pair.second && "Copies should be inserted exactly once.");
    return Ret;
  }

  // Try to coalesce a pair of vreg classes. CopyCls must be Copy's class.
  bool tryCoalesce(PhiCopy Copy, CongClass &CopyCls, CongClass &Other) {
    auto coalesce = [&]() {
      LLVM_DEBUG(dbgs() << "Coalescing " << *Copy.I << "\n");
      CopyCls.erase(*Copy.I);
      LocalNum.erase(Copy.I);
      ToClass.erase(Copy.I);
      Copy.I->replaceAllUsesWith(Copy.getSource());
      Copy.I->eraseFromParent();
    };
    if (&CopyCls == &Other) {
      // Possible when multiple affinities b/n same classes, e.g. if a phi had
      // the same incoming value for more than one pred.
      coalesce();
      return true;
    } else if (!intersects(CopyCls, Other)) {
      coalesce();
      mergeInto(CopyCls, Other);
      return true;
    }
    return false;
  }

  // Value copies should have already been inserted. Inserts operand copies and
  // coalesces phi-by-phi. TODO: Bug: Skip if BB is unreachable because
  // intersection check will not work.
  void insertAndCoalesce(BasicBlock &BB) {
    if (BB.empty() || !isa<PHINode>(BB.begin()))
      return;

    LLVM_DEBUG(dbgs() << "Examining phis of " << BB.getName() << "\n");
    DenseMap<const BasicBlock *, CongValue> Markers;
    for (BasicBlock *Pred : make_range(pred_begin(&BB), pred_end(&BB))) {
      // Accounts for multiple identical edges.
      auto Pair = Markers.insert({Pred, {}});
      if (Pair.second)
        Pair.first->second = createMarker(*Pred, None);
    }

    for (auto Pt = BB.begin(); Pt != BB.end() && isa<PHINode>(Pt); ++Pt) {
      auto &PN = *cast<PHINode>(Pt);
      // Insert operand copies.
      CongClass &PC = getCongClass(PN, None);
      for (auto &Pair : Markers) {
        CongValue &Copy = PC.add(insertCopy(*PN.getType(), Pair.second, None));
        ToClass.insert({Copy.I, &PC});
        for (unsigned OpNum = 0; OpNum < PN.getNumOperands(); OpNum += 1)
          if (PN.getIncomingBlock(OpNum) == Pair.first) {
            PhiCopy(*Copy.I).setSource(*PN.getOperand(OpNum));
            PN.setOperand(OpNum, Copy.I);
          }
        LLVM_DEBUG(dbgs() << "Inserted copy " << *Copy.I << "\n");
      }

      LLVM_DEBUG(dbgs() << "Fn is now " << F << "\n");

      std::sort(PC.Members.begin(), PC.Members.end(),
                [](const CongValue &A, const CongValue &B) {
                  return A.dpoBefore(B);
                });

      if (!CoalesceCopies)
        continue;

      // Coalesce operands.
      for (Use &U : make_range(PN.op_begin(), PN.op_end())) {
        LLVM_DEBUG(dbgs() << "Examining op " << *U << " of " << PN << "\n");
        // U would not be a copy if PN has multiple identical incomings.
        if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          PhiCopy Copy(*cast<IntrinsicInst>(U));
          if (auto *Src = dyn_cast<Instruction>(Copy.getSource()))
            if (tryCoalesce(Copy, PC, getCongClass(*Src, None)))
              LLVM_DEBUG(dbgs() << "Fn after coalescing: " << F << "\n");
        }
        // TODO: Handle constants and arguments as though they were defined in
        // entry.
      }

      // Coalesce value copy.
      assert(PN.hasOneUse());
      PhiCopy ValCopy(*cast<IntrinsicInst>(*PN.user_begin()));
      LLVM_DEBUG(dbgs() << "Examining value copy " << *ValCopy.I << "\n");
      if (tryCoalesce(ValCopy, getCongClass(*ValCopy.I, None), PC))
        LLVM_DEBUG(dbgs() << "Fn after coalescing: " << F << "\n");
    }
  }

  // TODO: This should be idempotent.
  void run() {
#if NVVM_VERSION < 700
    DT.DT->updateDFSNumbers();
#else
    DT.updateDFSNumbers();
#endif

    // Simulaneously local-number and insert value copies to ensure that phis
    // are fully isolated at insertAndCoalesce time.
    for (BasicBlock &BB : F) {
      auto It = BB.begin();
      for (; It != BB.end() && isa<PHINode>(It); ++It)
        localNumber(*It);
      if (It != BB.begin()) {
        // There were phis. Insert value copies.
        CongValue Mk = createMarker(BB, &*It);
        for (auto Pt = BB.begin(); Pt != BB.end() && isa<PHINode>(Pt); ++Pt) {
          auto &PN = *cast<PHINode>(Pt);
          CongValue ValCopy = insertCopy(*PN.getType(), Mk, &*It);
          getCongClass(*ValCopy.I, ValCopy);
          PN.replaceAllUsesWith(ValCopy.I);
          PhiCopy(*ValCopy.I).setSource(PN);
        }
      }
      for (; It != BB.end(); ++It)
        localNumber(*It);
    }
    LLVM_DEBUG(dbgs() << "Value copies inserted: " << F << "\n");

    for (BasicBlock &BB : F)
      insertAndCoalesce(BB);

    // Clean up useless pmarkers.
    for (IntrinsicInst *Marker : AllMarkers)
      if (Marker->use_empty())
        Marker->eraseFromParent();

    LLVM_DEBUG(dbgs() << "Minimal CSSA: " << F << "\n");
    LLVM_DEBUG(verifyFunction(F));
  }
};

struct CSSALegacyPass : public FunctionPass {
  static char ID;
  VisitFn VisitMembers;

  CSSALegacyPass() : CSSALegacyPass([](ArrayRef<Instruction *>) {}) {}

  CSSALegacyPass(VisitFn VFN) : FunctionPass(ID), VisitMembers(std::move(VFN)) {
    initializeCSSALegacyPassPass(*PassRegistry::getPassRegistry());
  }

#if NVVM_VERSION < 700
  const char *getPassName() const override { return PASS_NAME; }
#else
  StringRef getPassName() const override { return PASS_NAME; }
#endif

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MergeSetsWrapper>();
  }

  bool runOnFunction(Function &F) override {
#if NVVM_VERSION >= 700
    if (skipFunction(F))
      return false;
#endif
    if (DumpBefore)
      dbgs() << "IR Module before CSSA:\n" << *F.getParent() << "\n";
    CSSA Coal(F, getAnalysis<MergeSetsWrapper>().getMS());
    Coal.run();
    for (CongClass &CC : Coal.Classes) {
      if (!CC.size())
        continue;
      // TODO: Rework this.
      std::vector<Instruction *> Tmp;
      Tmp.reserve(CC.Members.size());
      for (const CongValue &CV : CC.Members)
        Tmp.push_back(CV.I);
      VisitMembers(Tmp);
    }
    return true;
  }
};

char CSSALegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsWrapper)
INITIALIZE_PASS_END(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)

Pass *nvvm::createCSSAPass() { return new CSSALegacyPass(); }

Pass *nvvm::createCSSAPass(VisitFn VFN) {
  return new CSSALegacyPass(std::move(VFN));
}
