set(LLVM_LINK_COMPONENTS
  MC
  Support
  XCoreInfo
  )

add_llvm_library(LLVMXCoreDesc
  XCoreInstPrinter.cpp
  XCoreMCTargetDesc.cpp
  XCoreMCAsmInfo.cpp
  )
