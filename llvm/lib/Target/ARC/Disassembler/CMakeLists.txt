add_llvm_library(LLVMARCDisassembler
  ARCDisassembler.cpp

  LINK_COMPONENTS

  ARCInfo
  MCDisassembler
  Support
  )
