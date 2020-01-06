#include "llvm/Transforms/Scalar/SparseDSE.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sparse-dse"

using namespace llvm;

raw_ostream &debug() { return dbgs() << DEBUG_TYPE << ": "; }

struct MemOp {
  unsigned DPONum;

  unsigned dpoNum() const { return; }
};

struct Stor : public MemOp {
  Instruction *I;
};

struct Kill : public MemOp {
  Instruction *I;
};

struct Lambda : public MemOp {
  BasicBlock *BB;
};

// Collect memory operations over an alloca.
struct AllocaMemOps : public InstVisitor<AllocaMemOps> {
  SmallVectorImpl<MemOp> &Yep;

  AllocaMemOps(SmallVectorImpl<MemOp> &Yep) : Yep(Yep) {}

  void collectUses(AllocaInst &AI) {
    SmallVector<Use *, 32> ToVisit;
    DenseSet<Use *> Visited{{AI}};

    auto pushUses = [&](User &UU) {
      // pre-condition: stack only contains unvisited uses.
      for (Use &U : UU.uses())
        // U could have been visited.
        if (Visited.insert(&U).second)
          // U must be unvisited.
          ToVisit.push_back(&U);
      // post-condition: preserved.
    };

    pushUses(AI);

    // pre-condition: stack only contains unvisited uses.
    while (!ToVisit.empty()) {
      Use *U = ToVisit.pop_back_val();
      if (auto *I = dyn_cast<Instruction>(U->getUser()))
        visit(*I);
      pushUses(*U);
    }
    // post-condition: preserved.
  }

  void visitLoadInst(LoadInst &LI) { Yep.push_back(Kill{&LI}); }

  void visitStoreInst(StoreInst &SI) { Yep.push_back(Stor{&SI}); }
};

PreservedAnalyses SparseDSEPass::run(Function &F, AnalysisManager<Function> &) {
  LLVM_DEBUG(debug() << "HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n");
  return PreservedAnalyses::all();
}
