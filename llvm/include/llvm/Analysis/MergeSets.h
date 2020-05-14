#ifndef LLVM_ANALYSIS_MERGESETS_H
#define LLVM_ANALYSIS_MERGESETS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Pass.h"

namespace llvm {
class raw_ostream;
class BasicBlock;
class Function;
class DominatorTree;
template <typename> class DomTreeNodeBase;
using DomTreeNode = DomTreeNodeBase<BasicBlock>;
class FunctionPass;

class MergeSets;

struct MergeInfo {
  unsigned BitPos;
  BitVector Merge;

  operator const BitVector &() const { return Merge; }

  BitVector selfUnion() const {
    BitVector Ret = *this;
    Ret.set(BitPos);
    return Ret;
  }

  raw_ostream &print(raw_ostream &, const MergeSets * = nullptr) const;
};

class MergeSets {
  const DominatorTree &DT;
  // The core reason for this analysis' existence. Record of each BB's merge
  // sets and canonical bit position (currently BFS order). Deliberately not
  // storing DomTreeNode as that could require an extra look-up from clients.
  DenseMap<const BasicBlock *, MergeInfo> Inner;
  // Used as a mapping from bit index to BB as well as iteration order for
  // TDMSC.
  std::vector<const DomTreeNode *> BfsVec;

  bool tdmsc();

  void computeBFS();

public:
  MergeSets(const DominatorTree &DT) : DT(DT) {
    computeBFS();
    for (bool AnotherPass = true; AnotherPass;)
      AnotherPass = tdmsc();
  }

  const DominatorTree &getDomTree() const { return DT; }

  // Returns nullptr if BB is unreachable.
  const MergeInfo *getMergeInfo(const BasicBlock &BB) const;

  const MergeInfo &getMergeInfo(const DomTreeNode &N) const;

  MergeInfo &getMergeInfo(const DomTreeNode &N);

  const DomTreeNode &atIndex(unsigned Idx) const { return *BfsVec[Idx]; }

  raw_ostream &print(raw_ostream &) const;
};

raw_ostream &operator<<(raw_ostream &O, const MergeSets &MS) {
  return MS.print(O);
}

struct MergeSetsWrapper : public FunctionPass {
  static char ID;
  std::unique_ptr<MergeSets> MS;

  MergeSetsWrapper();

  MergeSets &getMS() { return *MS; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &) override;
};

struct MergeSetsPrinterWrapper : public FunctionPass {
  static char ID;

  MergeSetsPrinterWrapper();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MergeSetsWrapper>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &) override;
};
} // end namespace llvm

#endif
