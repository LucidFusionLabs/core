Name "spaceball"
OutFile "spaceball-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\spaceball

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\spaceball.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\spaceball.lnk" "$INSTDIR\spaceball.exe"

SectionEnd
 
