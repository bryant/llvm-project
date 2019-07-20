set(LLVM_LINK_COMPONENTS
  MCDisassembler
  SparcInfo
  Support
  )

add_llvm_library(LLVMSparcDisassembler
  SparcDisassembler.cpp
  )
