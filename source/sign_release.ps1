param(
    [Parameter(Mandatory = $true)]
    [string]$CertPath,

    [Parameter(Mandatory = $true)]
    [string]$CertPassword,

    [string]$BundleRoot = ".\bundle\LensDiff.ofx.bundle\Contents\Win64"
)

$ErrorActionPreference = "Stop"

$signtool = Get-Command signtool.exe -ErrorAction Stop
$files = @(
    (Join-Path $BundleRoot "LensDiff.ofx")
)

foreach ($file in $files) {
    if (!(Test-Path $file)) {
        throw "Missing file to sign: $file"
    }

    & $signtool.Source sign `
        /f $CertPath `
        /p $CertPassword `
        /fd SHA256 `
        /tr http://timestamp.digicert.com `
        /td SHA256 `
        $file
}
