set(LLVM_LINK_COMPONENTS
  ARMDesc
  ARMInfo
  ARMUtils
  MCDisassembler
  Support
  )

add_llvm_library(LLVMARMDisassembler
  ARMDisassembler.cpp
  )
