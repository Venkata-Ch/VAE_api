<html>
<head><title></title>
 <style>
        body {
            margin: 3;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-end; 
            align-items: center; 
        }
        .image-format {
            display: flex;
            justify-content: center; 
            margin-bottom: 20px; 
        }
        .image-format img{
            margin: 0 15px;
        }
    </style>
</head>

<h3><strong>Variational Auto Encoder</strong></h3>
<body>
<p>The following tools has been built upon the following tools:<br>
1.PyTorch<br>
2.FastAPI<br>
3.Transformers<br>
4.Docker<br>
</p>

<p>A Variational AutoEncoder has been built using transformers
and trained on MNIST dataset using pytorch transformers.
The model is trained and evaluated according to Karpathy constant using BCELoss.
</p></body>

<p>You can run the docker image as follows:</p>
<code>docker run -d -p 5000:5000 vae_app:1 </code><br>


<video width="640" height="360" controls autoplay loop  playsinline>
  <source src="./assets/Screencast from 09-27-2024 10:19:33 AM.webm" type="video/webm">
</video>

<br><div class="image-format">  
<img src="/logos/Screenshot from 2024-09-25 14-40-13.png" style="vertical-align:middle" height="50" width="50"><br>     
<img src="/logos/docker-mark-blue.png" style="vertical-align:middle" height="60" width="60"></div>

</html>
