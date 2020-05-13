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
                                      Instruction *Ins) {
  IRBuilder<> IRB(Ins);
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

struct PhiCopy {
  IntrinsicInst *I;

  PhiCopy(Instruction &I) : I(*cast<IntrinsicInst>(I)) {
    assert(I->getIntrinsicID() == Intrinsic::nvvm_internal_copy);
  }

  Value *getSource() const { return I->getOperand(1); }

  Value *setSource(Value *V) const { return I->setOperand(1, V); }

  void coalesce() {
    LLVM_DEBUG(dbgs() << "Coalescing " << *I << "\n");
    I->replaceAllUsesWith(getSource());
    I->eraseFromParent();
  }
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

  // Returns true if this comes before Other in DPO. Mainly needed to maintain
  // sorted cong classes.
  bool dpoBefore(const CongValue &Other) const {
    return N->getDFSNumIn() <= Other.N->getDFSNumIn() ||
           LocalNum < Other.LocalNum;
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

  std::forward_list<CongClass> Classes;
  DenseMap<const Instruction *, CongClass *> ToClass;
  DenseMap<const Instruction *, unsigned> LocalNum;
  unsigned LocalCount = 0;

  CSSA(Function &F, const MergeSets &MS) : F(F), DT(MS.getDomTree()), MS(MS) {}

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
        return I->getParent() != &N.getParent() ||
               N.LocalNum < LocalNum.find(I)->second;
      });

    BitVector Mr = MS.getMergeInfo(N.getParent()).selfUnion();
    for (Use &U : Def.I->users()) {
      const BasicBlock *UB = getUseBlock(U);
      if (&N.getParent() == UB) {
        // Local dom checks. Note: This is the only time we have to worry about
        // pcopy nuances (since pcopy local num == pmarker local num)!
        if (isa<PHINode>(U.getUser()) ||
            N.LocalNum < LocalNum.find(U.getUser())->second)
          return true;
        // N !dom U, cannot be live from this use.
        continue;
      }

      assert(DT.dominates(Def.getNode(), DT.getNode(UB)) &&
             "How can Def not dominate use U?");
      for (const DomTreeNode *UN = DT.getNode(UB); UN != Def.getNode();
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
    LLVM_DEBUG(dbgs() << "Intersecting\n " << A << "\n  and" << B << "\n");
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
      std::pair<CongClass, bool> Cur;
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
    CongClass Merged;
    for (unsigned IA = 0, IB = 0; IA < A.size() || IB < B.size();) {
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

  // Find I's congruence class or else create one and return it. Specify CV to
  // save DT and LocalNum lookups.
  CongClass &getCongClass(Instruction &I, Optional<CongValue> CV) {
    auto Pair = ToClass.insert({&I, nullptr});
    if (Pair.second) {
      Classes.push_front(
          {{CV ? *CV
               : {&I, DT.getNode(I.getParent()), LocalNum.find(&I)->second}}});
      Pair.first->second = &Classes.front();
    }
    return *Pair.first->second;
  }

  unsigned localNumber(const Instruction &I) {
    return LocalNum.insert({&I, LocalCount++}).first->second;
  }

  // Insert and number a parallel copy marker.
  CongValue createMarker(BasicBlock &BB, Optional<Instruction *> Ins) {
    IntrinsicInst &Marker =
        createIntrinsic(Intrinsic::nvvm_internal_parallel_copy, {}, {},
                        Ins ? *Ins : BB->getTerminator());
    return {&Marker, DT.getNode(BB), localNumber(Marker)};
  }

  // Side concerns: Updating cong class with this inst; setting value op.
  CongValue insertCopy(Type &Ty, const CongValue &Marker,
                       Optional<Instruction *> Ins) {
    CongValue Ret = Marker;
    Ret.I = &createIntrinsic(Intrinsic::nvvm_internal_copy, {&Ty},
                             {&Marker.getInst(), UndefValue::get(&Ty)},
                             Ins ? *Ins : Marker.getParent().getTerminator());
    LocalNum.insert({Ret.I, Ret.LocalNum});
    return Ret;
  }

  // Try to coalesce a pair of vreg classes. CopyCls must be Copy's class.
  bool tryCoalesce(PhiCopy Copy, CongClass &CopyCls, CongClass &Other) {
    if (&CopyCls == &Other) {
      // Possible when multiple affinities b/n same classes, e.g. if a phi had
      // the same incoming value for more than one pred.
      Copy.coalesce();
      CopyCls.erase(Copy);
    } else if (!intersects(CopyCls, Other)) {
      Copy.coalesce();
      CopyCls.erase(Copy);
      mergeInto(CopyCls, Other);
    }
  }

  // Value copies should have already been inserted. Inserts operand copies and
  // coalesces phi-by-phi. TODO: Bug: Skip if BB is unreachable because
  // intersection check will not work.
  void insertAndCoalesce(BasicBlock &BB) {
    if (!BB.getFirstNonPHI())
      return;

    DenseMap<const BasicBlock *, CongValue> Markers;
    Markers.reserve(std::distance(BB.pred_begin(), BB.pred_end()));
    for (BasicBlock &Pred : make_range(BB.pred_begin(), BB.pred_end())) {
      // Accounts for multiple identical edges.
      auto Pair = Markers.insert({&Pred, {}});
      if (Pair.second)
        Pair.first->second = createMarker(Pred, None);
    }

    for (PHINode &PN : BB.phis()) {
      // Insert operand copies.
      CongClass &PC = Coalescer.getCongClass(PN, None);
      for (Use &U : PN.operands()) {
        CongValue &Copy = PC.add(insertCopy(*PN.getType(), Marker, None));
        PhiCopy(Copy.I).setSource(U.get());
        U.setCopy(I);
      }

      PC.sort();

      // Coalesce operands.
      for (Use &U : PN.operands()) {
        PhiCopy Copy(*cast<IntrinsicInst>(U));
        tryCoalesce(Copy, PC, getCongClass(Copy.getSource(), None));
      }

      // Coalesce value copy.
      assert(PN.hasOneUse());
      PhiCopy ValCopy(*cast<IntrinsicInst>(PN.user_begin()));
      tryCoalesce(ValCopy, getCongClass(ValCopy, None), PC);
    }

    // TODO: Clean up useless pmarkers.
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
    for (const BasicBlock &BB : F) {
      auto It = BB.begin();
      for (; It != BB.end() && isa<PHINode>(It); ++It)
        localNumber(It);
      if (It != BB.begin()) {
        // There were phis. Insert value copies.
        CongValue Mk = createMarker(BB, It);
        for (PHINode &PN : BB.phis()) {
          CongValue ValCopy = insertCopy(*PN.getType(), Mk, It);
          // TODO: Enhancement: Include a reference to val copy CV from phi CC
          // to save cost of one lookup.
          getCongClass(ValCopy.I, ValCopy);
          PN.replaceAllUsesWith(ValCopy.I);
          PhiCopy(ValCopy).setSource(PN);
        }
      }
      for (; It != BB.end(); ++It)
        localNumber(It);
    }

    for (BasicBlock &BB : F)
      insertAndCoalesce(BB);

    LLVM_DEBUG(dbgs() << "CSSA " << F.getName() << "\n" << F << "\n");
    verifyFunction(F);
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

  // TODO: Bug: Maybe require unreachables to be pruned.
  bool runOnFunction(Function &F) override {
#if NVVM_VERSION >= 700
    if (skipFunction(F))
      return false;
#endif
    return CSSA(F, getAnalysis<MergeSetsPass>().getMS()).run();
  }
};

char CSSALegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsPass)
INITIALIZE_PASS_END(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)

Pass *nvvm::createCSSAPass() { return new CSSALegacyPass(); }
