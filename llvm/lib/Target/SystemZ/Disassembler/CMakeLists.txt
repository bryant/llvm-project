set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  SystemZDesc
  SystemZInfo
  )

add_llvm_library(LLVMSystemZDisassembler
  SystemZDisassembler.cpp
  )
