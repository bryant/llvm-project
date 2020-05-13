// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END

// Computes the merge sets (really just the IDF) of each basic block, but with
// average complexity that is linear to number of blocks.

#define DEBUG_TYPE "merge-sets"
#define PASS_NAME "Merge sets"

#include "llvm/Analysis/MergeSets.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

struct PrintBV {
  const BitVector &M;
  const MergeSets *MS;
  PrintBV(const BitVector &M, const MergeSets *MS = nullptr) : M(M), MS(MS) {}

  friend raw_ostream &operator<<(raw_ostream &O, const PrintBV &MV) {
    for (unsigned BitPos : MV.M.set_bits())
      if (MV.MS)
        O << " " << &MV.MS->atIndex(BitPos);
      else
        O << " " << BitPos;
    return O;
  }
};

namespace llvm {
// Abbreviated dom tree node printouts.
raw_ostream &operator<<(raw_ostream &O, const DomTreeNode &N) {
  return O << N.getBlock()->getName();
}
} // namespace llvm

// Iterate over N's j-predecessors.
struct JPredIter : const_pred_iterator {
  const DomTreeNode &N;
  const DomTreeNode *NextPred = nullptr;
  const DominatorTree *DT = nullptr;

  JPredIter(const DomTreeNode &N, const DominatorTree &DT)
      : const_pred_iterator(N.getBlock()), N(N), DT(&DT) {
    advancePastNonTerminators();
  }

  JPredIter(const DomTreeNode &N)
      : const_pred_iterator(N.getBlock(), false), N(N) {}

  inline void advancePastNonTerminators() {
    while (!It.atEnd()) {
      if (auto *Inst = dyn_cast<Instruction>(*It))
        if (Inst->isTerminator())
          if (const DomTreeNode *Next = DT->getNode(Inst->getParent()))
            if (!DT->dominates(Next, &N)) {
              NextPred = Next;
              break;
            }
      ++It;
    }
  }

  inline const DomTreeNode &operator*() const { return *NextPred; }
};

iterator_range<JPredIter> jpreds(const DomTreeNode &N,
                                 const DominatorTree &DT) {
  return make_range(JPredIter(N, DT), JPredIter(N));
}

// Computes BfsVec and build bitpos/node mappings.
void MergeSets::computeBFS() {
  BfsVec.reserve(DT.size());
  BfsVec.push_back(DT.getRootNode());
  for (unsigned I = 0; I < BfsVec.size(); I += 1) {
    std::copy(BfsVec[I]->begin(), BfsVec[I]->end(), std::back_inserter(BfsVec));
    Inner.insert({BfsVec[I]->getBlock(), {I, BitVector(DT.size())}});
  }
}

const DomTreeNode &propagateToShadow(const DomTreeNode &P, const DomTreeNode &S,
                                     MergeSets &MS) {
  BitVector Update = MS.getMergeInfo(S).selfUnion();

  // merge(l) |= merge(s) \union s
  auto prop = [&](const DomTreeNode &L) {
    LLVM_DEBUG(dbgs() << "Propagating " << L << " |= " << S << "\n");
    MS.getMergeInfo(L).Merge |= Update;
  };

  const DomTreeNode *L;
  for (L = &P; L->getIDom() != S.getIDom(); L = L->getIDom())
    prop(*L);
  prop(*L);
  return *L;
}

// TODO: Impl and bench TDMSC 2.
// TODO: Impl and bench with post-ordered dom tree levels.
template <typename V>
static bool checkInconsistency1(const DomTreeNode &L, const MergeSets &MS,
                                V &&Visited) {
  const BitVector &MergeL = MS.getMergeInfo(L);
  for (const DomTreeNode &P : jpreds(L, MS.getDomTree())) {
    if (Visited.count({&P, &L})) {
      // If consistent, merge(P) should contain merge(L), so check ~P & L.
      BitVector MergeP = MS.getMergeInfo(P);
      LLVM_DEBUG(dbgs() << "merge(" << P << ") =" << PrintBV(MergeP)
                        << " should contain merge(" << L
                        << ") =" << PrintBV(MergeL) << "\n");
      MergeP.flip();
      MergeP &= MergeL;
      if (MergeP.any())
        return true;
    } else
      LLVM_DEBUG(dbgs() << "Skipping unvisited (" << P << ", " << L << ")\n");
  }
  return false;
}

bool MergeSets::tdmsc() {
  // Visited edges.
  DenseSet<std::pair<const DomTreeNode *, const DomTreeNode *>> Visited;
  bool Inconsistent = false;

  for (const DomTreeNode *N : BfsVec) {
    for (const DomTreeNode &P : jpreds(*N, getDomTree())) {
      if (!Visited.insert({&P, N}).second)
        continue;
      LLVM_DEBUG(dbgs() << "Visiting j-edge (" << P << ", " << *N << ")\n");
      const DomTreeNode &L = propagateToShadow(P, *N, *this);
      Inconsistent = checkInconsistency1(L, *this, Visited);
      LLVM_DEBUG(dbgs() << "Inconsistent? " << Inconsistent << "\n");
    }
  }

  return Inconsistent;
}

const MergeInfo *MergeSets::getMergeInfo(const BasicBlock &BB) const {
  assert(Inner.count(&BB) ||
         !DT.isReachableFromEntry(&BB) && "Forgot to recompute over this fn?");
  auto It = Inner.find(&BB);
  return It == Inner.end() ? nullptr : &It->second;
}

const MergeInfo &MergeSets::getMergeInfo(const DomTreeNode &N) const {
  return Inner.find(N.getBlock())->second;
}

MergeInfo &MergeSets::getMergeInfo(const DomTreeNode &N) {
  return Inner.find(N.getBlock())->second;
}

raw_ostream &MergeInfo::print(raw_ostream &O, const MergeSets *MS) const {
  if (MS)
    O << MS->atIndex(BitPos);
  else
    O << BitPos;
  return O << ":" << PrintBV(Merge, MS) << "\n";
}

raw_ostream &MergeSets::print(raw_ostream &O) const {
  O << "Merge sets for fn "
    << DT.getRootNode()->getBlock()->getParent()->getName() << ":\n";
  for (const auto &Pair : Inner)
    Pair.second.print(O << "  ", this);
  return O;
}

MergeSetsWrapper::MergeSetsWrapper() : FunctionPass(ID) {
  initializeMergeSetsWrapperPass(*PassRegistry::getPassRegistry());
}

void MergeSetsWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesAll();
}

bool MergeSetsWrapper::runOnFunction(Function &) {
  MS.reset(new MergeSets(getAnalysis<DominatorTreeWrapperPass>().getDomTree()));
  return false;
}

char MergeSetsWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(MergeSetsWrapper, DEBUG_TYPE, PASS_NAME, true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(MergeSetsWrapper, DEBUG_TYPE, PASS_NAME, true, true)

MergeSetsPrinterWrapper::MergeSetsPrinterWrapper() : FunctionPass(ID) {
  initializeMergeSetsPrinterWrapperPass(*PassRegistry::getPassRegistry());
}

bool MergeSetsPrinterWrapper::runOnFunction(Function &) {
  dbgs() << getAnalysis<MergeSetsWrapper>().getMS();
  return false;
}

char MergeSetsPrinterWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(MergeSetsPrinterWrapper, "print-" DEBUG_TYPE,
                      PASS_NAME " printer", false, false)
INITIALIZE_PASS_DEPENDENCY(MergeSetsWrapper)
INITIALIZE_PASS_END(MergeSetsPrinterWrapper, "print-" DEBUG_TYPE,
                    PASS_NAME " printer", false, false)
