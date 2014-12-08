Name "browser"
OutFile "browser-installer.exe"
InstallDir $PROGRAMFILES\com.lucidfusionlabs\browser

Page directory
Page instfiles

Section "";

RmDir /r "$INSTDIR"

SetOutPath "$INSTDIR"
File "Debug\browser.exe"
File "Debug\*.dll"

SetOutPath "$INSTDIR\assets"
File "assets\*"

SetOutPath "$INSTDIR"
CreateDirectory "$SMPROGRAMS\com.lucidfusionlabs"
createShortCut "$SMPROGRAMS\com.lucidfusionlabs\browser.lnk" "$INSTDIR\browser.exe"

SectionEnd
 
