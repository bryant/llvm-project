// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Transforms a given function into CSSA form, optionally coalescing copies
// where possible.

#define DEBUG_TYPE "do-cssa"
#define PASS_NAME "Insert phi elim copies"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

// TODO: Hook this up to some part of ClientAPI, probably NVVMCodeGenOptions.
static cl::opt<bool> CoalesceCopies("cssa-coalesce" cl::init(true), cl::Hidden);

struct Copy {
  IntrinsicInst *I;
};

static IntrinsicInst &insertCopyImpl(ArrayRef<Type *> Tys,
                                     ArrayRef<Value *> Vals, IRBuilder<> &IRB) {
  return *cast<IntrinsicInst>(
      IRB.CreateIntrinsic(Intrinsic::nvvm_internal_copy, Tys, Vals));
}

template <typename T>
static IntrinsicInst &insertCopy(PHINode &P, unsigned OpNum, Value *PCopy,
                                 T &&IRB) {
  auto &Copy = insertCopyImpl({P.getType(), PCopy->getType(), P.getType()},
                              {PCopy, P.getIncomingValue(OpNum)}, IRB);
  P.setIncomingValue(OpNum, &Copy);
  return Copy;
}

struct PhiCopy {
  IntrinsicInst *I;
};

struct CongValue {
  Instruction *I;
  unsigned DFSIn = 0;
  unsigned DFSOut = 0;
  unsigned DPONum = 0;
};

struct CongClass {
  // This should always be sorted by DPO.
  std::vector<CongValue> Members;

  CongValue &operator[](unsigned Idx) { return Members[Idx]; }

  const CongValue &operator[](unsigned Idx) const { return Members[Idx]; }
};

// TODO: CRITICAL: Need to init DPO numbering, CongClasses for split phis.
struct CSSA {
  Function &F;
  const DominatorTree &DT;
  const MergeSets &MS;
  // Un-coalesced copies.
  std::vector<PhiCopy> Copies;
  std::vector<PhiCopy> Coalescable;

  std::forward_list<CongClass> Classes;
  DenseMap<Instruction *, CongClass *> ToClass;

  CSSA(Function &F, const MergeSets &MS) : F(F), DT(MS.getDomTree()), MS(MS) {}

  // Is Def live at N?
  bool isLiveIn(const CongValue &N, const CongValue &Def) const {
    assert(&N != &Def && Def.dom(N) &&
           "Should be part of the Budimlic dom-forest check.");

    // Edge cases: 1) nb == db; 2) nb == ub where u is a use of def.

    if (N.getBlock() == Def.getBlock())
      return any_of(Def.users(), [&](const User &U) { return &U == N.I; });

    const DomTreeNode *DB = DT.getNode(Def.getParent());
    BitVector Mr = MS.getMergeBitVec(N.getParent());
    Mr.set(MS.getIndex(N.getParent()));

    for (const User &U : Def.users())
      for (const DomTreeNode *UB = DT.getNode(U.getParent()); UB != DB;
           UB = UB->getIDom())
        if (Mr.test(MS.getIndex(*UB))) {
          // Some block in merge(N) + {N} doms U. Account for nb == ub.
          if (U.getParent() != &N.getParent() || DT.dominates(N.I, &U))
            return true;
        }

    return false;
  }

  // Given intersection-free sets A and B, decide if the combination of both
  // self-intersects.
  bool intersects(const CongClass &A, const CongClass &B) const {
    // TODO: This does Budimlic checks over A merged with B where both are
    // DPO-ordered, but we could also do the actual merge here to save an extra
    // iter. May not be worthwhile, though.

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
      StackEntry Cur =
          ((IA < A.size() && IB < B.size() && A[IA] < B[IB]) || IB >= B.size())
              ? {A[IA++], StackEntry::FromA}
              : {B[IB]++, StackEntry::FromB};

      while (!Stack.empty() && !Stack.back().V.dom(Cur.V))
        Stack.pop_back();

      // Skip checking if TOS and Cur came from the same class since the fn
      // pre-condition already ensures no intersection.
      if (!Stack.empty() && Stack.back().Cls != Cur.Cls &&
          isLiveIn(Stack.back(), Cur))
        return true;

      Stack.push_back(Cur);
    }
    return false;
  }

  void mergeInto(CongClass &A, CongClass &B) {
    // TODO: Optimize for the common case when one of A or B are singleton.
    CongClass Merged;
    for (unsigned IA = 0, IB = 0; IA < A.size() && IB < B.size();) {
      if (A[IA] >= B[IB]) {
        CongValue Cur = B[IB++];
        auto It = ToClass.find(Cur.I);
        assert(It != ToClass.end() &&
               "Value-to-class out of sync from class-to-value.");
        It->second = &A;
        Merged.Members.push_back(Cur);
      } else
        Merged.Members.push_back(A[IA++]);
    }
    std::swap(Merged.Members, A);
    B.Members.clear();
  }

  // Find I's congruence class or else create one and return it.
  CongClass &getCongClass(Instruction &I) {
    auto Pair = ToClass.insert({&I, nullptr});
    if (Pair.second) {
      Classes.push_front({{CongValue::fromInst(I)}});
      Pair.first->second = &Classes.front();
    }
    return *Pair.first->second;
  }

  CSSA &coalesceAllCopies() {
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (unsigned Idx = 0; Idx < Copies.size();) {
        // TODO: Handle constants and arguments as though they were defined in
        // entry.
        if (!isa<Instruction>(Copies[Idx].getSource()))
          continue;

        // TODO: Optimize for the common case when one of CC or SrcCC are
        // singleton.
        CongClass &CC = getCongClass(*Copies[Idx].I);
        CongClass &SrcCC = getCongClass(Copies[Idx].getSource());
        if (!intersects(CC, SrcCC)) {
          // No interference, can coalesce.
          std::swap(Copies.back(), Copies[Idx]);
          Coalescable.push_back(Copies.pop_back());

          mergeInto(CC, SrcCC);

          Changed = true;
        } else
          Idx += 1;
      }
    }
    return *this;
  }

  void insertPhiCopies(BasicBlock &BB) {
    if (BB.empty() || !isa<PHINode>(BB.begin()))
      return;

    // Insert operand copies.
    auto &P0 = *cast<PHINode>(BB.begin());
    for (unsigned PredNum = 0; PredNum < P0.getNumIncomingValues();
         PredNum += 1) {
      BasicBlock *Pred = P0.getIncomingBlock(PredNum);
      IRBuilder<> IRB(Pred->getTerminator());
      CallInst *Parallel = IRB.CreateIntrinsic(
          Intrinsic::nvvm_internal_parallel_copy, None, None);

      insertCopy(P0, PredNum, Parallel, IRB);
      for (auto It = std::next(BB.begin()); It != BB.end() && isa<PHINode>(It);
           ++It)
        insertCopy(cast<PHINode>(*It), PredNum, Parallel, IRB);
    }

    // Break phi live ranges.
    IRBuilder<> IRB(
        &*find_if(BB, [](const Instruction &I) { return !isa<PHINode>(I); }));
    CallInst &P0Copy = *IRB.CreateIntrinsic(Intrinsic::ssa_copy, {P0.getType()},
                                            {UndefValue::get(P0.getType())});
    P0.replaceAllUsesWith(&P0Copy);
    P0Copy.setOperand(0, &P0);
    for (auto It = std::next(BB.begin()); It != BB.end() && isa<PHINode>(It);
         ++It) {
      CallInst &P_ = *IRB.CreateIntrinsic(Intrinsic::ssa_copy, {It->getType()},
                                          {UndefValue::get(It->getType())});
      It->replaceAllUsesWith(&P_);
      P_.setOperand(0, &*It);
    }
  }

  // TODO: This should be idempotent.
  CSSA &insertCopies() {
    for (BasicBlock &BB : F)
      insertPhiCopies(BB);
    return *this;
  }
};

struct CSSALegacyPass : public FunctionPass {
  static char ID;

  CSSALegacyPass() : FunctionPass(ID) {
    initializeCSSAPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MergeSetsWrapperPass>();
  }

  bool runOnFunction(Function &F) const override {
    if (skipFunction(F))
      return false;
    return CSSA(F, getAnalysis<MergeSetsWrapperPass>().getMS())
        .insertCopies()
        .coalesceAllCopies();
  }
};

char CSSALegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsWrapperPass)
INITIALIZE_PASS_END(CSSALegacyPass, DEBUG_TYPE, PASS_NAME, false, false)

Pass *llvm::createCSSAPass() { return new CSSALegacyPass(); }
