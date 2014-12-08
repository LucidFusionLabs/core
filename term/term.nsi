Name "term"
OutFile "term-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\term

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\term.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\term.lnk" "$INSTDIR\term.exe"

SectionEnd
 
