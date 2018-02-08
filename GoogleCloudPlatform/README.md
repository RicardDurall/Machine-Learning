<h1>Installing NVIDIA CUDA on Google Cloud Platform with Tesla K80 and Ubuntu 16.04</h1>
<h2>Create Virtual Machine</h2>
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
Congrats VM created!!
<p>Now let's get and static IP and set a new firewall rule:
  <br>First go to:
  <br>VPC network > External IP addresses 
  <br>asign one IP to your VM
  <br>Then go to:
  <br>VPC network > Firewall rule details
  <br>and copy the values described on the picture:
</p>
<img src="https://github.com/RicardDurall/Machine-Learning/tree/master/GoogleCloudPlatform/Images/Selection_003.png">
