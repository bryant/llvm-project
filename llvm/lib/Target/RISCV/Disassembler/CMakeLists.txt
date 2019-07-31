add_llvm_library(LLVMRISCVDisassembler
  RISCVDisassembler.cpp

  LINK_COMPONENTS

  MCDisassembler
  RISCVInfo
  Support
  )
