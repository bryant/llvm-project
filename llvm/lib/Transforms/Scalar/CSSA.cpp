#define DEBUG_TYPE "do-cssa"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

static cl::opt<bool> CoalesceCopies("cssa-coalesce" cl::init(true), cl::Hidden);

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

static void insertAllCopies(Function &F) {
  for (BasicBlock &BB : F) {
    if (BB.empty() || !isa<PHINode>(BB.begin()))
      continue;
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
}

struct CSSAPass : public FunctionPass {
  static char ID;

  CSSAPass() : FunctionPass(ID) {
    initializeCSSAPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    insertAllCopies(F);

    if (CoalesceCopies)
      AggCoalescer(F);

    return true;
  }
};

char CSSAPass::ID = 0;

INITIALIZE_PASS_BEGIN(CSSAPass, DEBUG_TYPE, "", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CSSAPass, DEBUG_TYPE, "Insert SSA copies for phi elim.",
                    false, false)

Pass *llvm::createCSSAPass() { return new CSSAPass(); }
