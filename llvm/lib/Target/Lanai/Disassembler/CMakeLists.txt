set(LLVM_LINK_COMPONENTS
  LanaiDesc
  LanaiInfo
  MC
  MCDisassembler
  Support
  )

add_llvm_library(LLVMLanaiDisassembler
  LanaiDisassembler.cpp
  )
