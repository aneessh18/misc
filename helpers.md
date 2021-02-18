### Important thing to remember while opening websockets in AWS EC2
while creating a web socket server the host address should be filled with the instance private IP address.
But while calling the server from the browser, its public IP should be used
AWS maps the public IP to the private IP internally 
