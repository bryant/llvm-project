set(LLVM_LINK_COMPONENTS
  MCDisassembler
  Support
  XCoreInfo
  )

add_llvm_library(LLVMXCoreDisassembler
  XCoreDisassembler.cpp
  )
