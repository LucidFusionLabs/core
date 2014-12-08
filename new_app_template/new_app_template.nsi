Name "$BINNAME"
OutFile "$BINNAME-installer.exe"
InstallDir $PROGRAMFILES\$ORGNAME\$BINNAME

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\$BINNAME.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\$ORGNAME"
createShortCut "$SMPROGRAMS\$ORGNAME\$BINNAME.lnk" "$INSTDIR\$BINNAME.exe"

SectionEnd
 
