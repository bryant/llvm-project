set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(LLVMMSP430Info
  MSP430TargetInfo.cpp
  )
