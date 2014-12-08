Name "calculator"
OutFile "calculator-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\calculator

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\calculator.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\calculator.lnk" "$INSTDIR\calculator.exe"

SectionEnd
 
