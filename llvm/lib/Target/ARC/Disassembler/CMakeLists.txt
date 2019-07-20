set(LLVM_LINK_COMPONENTS
  ARCInfo
  MCDisassembler
  Support
  )

add_llvm_library(LLVMARCDisassembler
  ARCDisassembler.cpp
  )
