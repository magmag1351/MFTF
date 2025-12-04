$baseUrl = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
$lbfUrl = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

Write-Host "Downloading model files..."

Invoke-WebRequest -Uri $baseUrl -OutFile "haarcascade_frontalface_alt2.xml"
Write-Host "Downloaded haarcascade_frontalface_alt2.xml"

Invoke-WebRequest -Uri $lbfUrl -OutFile "lbfmodel.yaml"
Write-Host "Downloaded lbfmodel.yaml"

Write-Host "Done."
