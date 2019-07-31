add_llvm_library(LLVMSystemZDesc
  SystemZInstPrinter.cpp
  SystemZMCAsmBackend.cpp
  SystemZMCAsmInfo.cpp
  SystemZMCCodeEmitter.cpp
  SystemZMCObjectWriter.cpp
  SystemZMCTargetDesc.cpp

  LINK_COMPONENTS

  MC
  Support
  SystemZInfo
  )
