<h1>Create Virtual Machine and its dependencies</h1>
<p>First of all create an account on Google Clous Platform.
  <br>Then go to:
  <br>Compute Engine > VM Instances > Create Instance
  <br>and follow the steps described on the pictures:
</p>
<img src="https://github.com/RicardDurall/Machine-Learning/tree/master/GoogleCloudPlatform/Images/Selection_001.png">
 <ul>
  <li>Choose a name that you like (e.g. test)</li>
  <li>Then select zone (us-east1-b is the cheapest one)</li>
  <li>Number of CPUs, memory and GPUs according to your needs</li>
</ul> 
<img src="https://github.com/RicardDurall/Machine-Learning/tree/master/GoogleCloudPlatform/Images/Selection_002.png">
 <ul>
  <li>Select Ubuntu 16.04 with 40GB space</li>
  <li>Allow both firewall checkboxes</li>
  <li>Add yout SSH public key if you have one (Optional)</li>
</ul>
<p>Now let's get and static IP and set a new firewall rule:
  <br>First go to:
  <br>VPC network > External IP addresses 
  <br>asign one IP to your VM
  <br>Then go to:
  <br>VPC network > Firewall rule details
  <br>and copy the values described on the picture:
</p>
<img src="https://github.com/RicardDurall/Machine-Learning/tree/master/GoogleCloudPlatform/Images/Selection_003.png">

Congrats your VM is ready to be used!!
<h1>Installing NVIDIA CUDA on Google Cloud Platform with Tesla K80 and Ubuntu 16.04</h1>
<h3>Technical Specifications</h3>
 <ul>
  <li>NVIDIA driver 375.66</li>
  <li>CUDA Toolkit 8.0</li>
  <li>cuDNN 5.1 </li>
</ul>
Tested with 1x Tesla K80.
<pre> lspci | grep -i NVIDIA</pre>

<h3>NVIDIA drivers</h3>
We will install the NVIDIA Tesla Driver via deb package.
<pre>wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
  <br>sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
  <br>sudo apt-get update
  <br>sudo apt-get install cuda-drivers 
</pre>
<h3>CUDA toolkit</h3>
<pre>wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  <br>sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  <br>sudo apt-get update
  <br>sudo apt-get install cuda-8.0
</pre>
<h3>cuDNN</h3>
