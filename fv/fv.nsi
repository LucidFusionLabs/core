Name "fv"
OutFile "fvinst.exe"
InstallDir $PROGRAMFILES\lfl\fv

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\fv.exe"
File "Debug\*.dll"
File "README.txt"
File "GPL.txt"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\LFL"
createShortCut "$SMPROGRAMS\LFL\fv.lnk" "$INSTDIR\fv.exe"

SectionEnd
 