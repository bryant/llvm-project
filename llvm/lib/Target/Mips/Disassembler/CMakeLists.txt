set(LLVM_LINK_COMPONENTS
  MCDisassembler
  MipsInfo
  Support
  )

add_llvm_library(LLVMMipsDisassembler
  MipsDisassembler.cpp
  )
