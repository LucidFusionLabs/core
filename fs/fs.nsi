Name "fs"
OutFile "fsinst.exe"
InstallDir $PROGRAMFILES\lfl\fs

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\fs.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

ExecWait "$INSTDIR\fs -install"

SectionEnd
 