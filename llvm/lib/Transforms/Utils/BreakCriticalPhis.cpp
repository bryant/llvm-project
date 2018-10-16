// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// NVIDIA_COPYRIGHT_END

// \file Splits critical phi edges only where needed. This is primarily needed
// in D2IR to ensure that vreg copies (from phi lowering) are post-dominated by
// the lowered phi block (otherwise, there would be a path from the copy inst to
// exit that doesn't cross the phi, IE unnecessary copy).
//
// If there's no phi, the critical edge won't matter and we don't split to avoid
// unnecessary jumps in ORI.
//
// If the critical edge happens to be a loop back edge, we really don't want to
// split because this de-canonicalizes the loop (if it were already rotated) so
// that re-canonicalization is possible and thus inhibits further OCG loop opts.
//
// TODO: Hook this up to the new pass manager and/or expose to opt in order to
// unit-test properly.

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BreakCriticalPhis.h"

#define DEBUG_TYPE "break-crit-phis"

using namespace llvm;

// Shorthand to print an edge to raw_ostream.
struct edge {
  const BasicBlock *From, *To;
  edge(const BasicBlock &F, const BasicBlock &T) : From(&F), To(&T) {}
  friend raw_ostream &operator<<(raw_ostream &O, const edge &E) {
    return O << E.From->getName() << ", " << E.To->getName();
  }
};

struct BreakCriticalPhis : public FunctionPass {
  static char ID;

  BreakCriticalPhis() : FunctionPass(ID) {
    initializeBreakCriticalPhisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We need LoopInfo for back edge info.
    AU.addRequired<LoopInfoWrapperPass>();
    // We need DT because SplitCriticalEdge requires updating both LoopInfo and
    // DT or none at all. This should not incur additional cost because LoopInfo
    // itself already depends on DT.
    AU.addRequired<DominatorTreeWrapperPass>();

    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreservedID(LoopSimplifyID);
  }
};

char BreakCriticalPhis::ID = 0;

INITIALIZE_PASS_BEGIN(BreakCriticalPhis, "break-crit-phis",
                      "D2IR Break Critical Edges", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(BreakCriticalPhis, "break-crit-phis",
                    "D2IR Break Critical Edges", false, false)

FunctionPass *llvm::createBreakCriticalPhisPass() {
  return new BreakCriticalPhis();
}

static bool hasCriticalPhis(const BasicBlock &, const LoopInfo &);

bool BreakCriticalPhis::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  bool Changed = false;

  // Look at phi-containing succs of multiple-succ blocks. If there are no phis
  // in the critical edge succ, then we never need to split. If the phi is
  // useless (succ only has one pred), SplitCriticalEdge will catch this and
  // refuse to split, which is also okay.
  for (BasicBlock &BB : F) {
    TerminatorInst &TI = *BB.getTerminator();
    if (TI.getNumSuccessors() <= 1 || isa<IndirectBrInst>(&TI))
      continue;

    Loop *BBL = LI.getLoopFor(&BB);

    for (unsigned SuccIdx = 0; SuccIdx < TI.getNumSuccessors(); SuccIdx += 1) {
      // Figure out if (BB, Succ) is a critical edge that should be split.

      BasicBlock &Succ = *TI.getSuccessor(SuccIdx);
      auto IsBackEdge = [&]() { return BBL && BBL->getHeader() == &Succ; };

      if (!Succ.empty() && isa<PHINode>(Succ.front()) && !IsBackEdge()) {
        LLVM_DEBUG(dbgs() << "Attempting to split (" << edge(BB, Succ)
                          << ")\n");

        CriticalEdgeSplittingOptions Opts(&DT, &LI);
        Opts.setMergeIdenticalEdges();

        if (BasicBlock *NewNode = SplitCriticalEdge(&TI, SuccIdx, Opts)) {
          LLVM_DEBUG(dbgs() << "Split node is " << NewNode->getName() << "\n");
          Changed = true;
        }
      } else if (IsBackEdge())
        LLVM_DEBUG(dbgs() << "Skipping back edge (" << edge(BB, Succ) << ")\n");
    }
  }

  // TODO: (NVVM60) llvm::all_of shorthand.
  assert(std::all_of(
             F.begin(), F.end(),
             [&](const BasicBlock &BB) { return !hasCriticalPhis(BB, LI); }) &&
         "Discovered critical phi edge.");

  return Changed;
}

// Return true iff BB has at least one phi, multiple preds, and at least one
// multiple-succ pred.
static bool hasCriticalPhis(const BasicBlock &BB, const LoopInfo &LI) {
  if (BB.empty() || !isa<PHINode>(BB.front()) || BB.isEHPad())
    return false;

  const Loop *BBL = LI.getLoopFor(&BB);

  // TODO: (NVVM60) llvm::predecessors() shorthand.
  for (const BasicBlock *Pred : make_range(pred_begin(&BB), pred_end(&BB))) {
    if (BBL && BBL->getHeader() == &BB && LI.getLoopFor(Pred) == BBL)
      // Was a back edge. Of course we did not split.
      continue;

    if (!isa<IndirectBrInst>(Pred->getTerminator()) &&
        isCriticalEdge(Pred->getTerminator(), GetSuccessorNumber(Pred, &BB),
                       /* MergeIdenticalEdges = */ true)) {
      LLVM_DEBUG(dbgs() << "Critical phi (" << edge(*Pred, BB) << ")\n");
      return true;
    }
  }

  return false;
}
