add_llvm_library(LLVMBPFDisassembler
  BPFDisassembler.cpp

  LINK_COMPONENTS

  BPFInfo
  MCDisassembler
  Support
)

