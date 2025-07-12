---
title: "Mastering AWS EC2"
description: "MediaVault Deep Dive: Mastering AWS EC2 for Scalable Applications"
date: "Jul 14 2025"
---

Welcome back to the MediaVault series! So far, we've laid out the project's vision and secured our account with IAM. While MediaVault isn't running on a server just yet, the next logical step is to provision some compute power. To do that, I'll be turning to the workhorse of AWS: **Amazon Elastic Compute Cloud (EC2)**.

This post outlines the approach I'll take when it's time to launch our first virtual server, including the crucial decision of what kind of storage to use. Planning this out now is key for a smooth, secure, and cost-effective deployment.

---

### What is an EC2 Instance? 🖥️

Think of an **EC2 instance** as a virtual server in the cloud. It's your own slice of a computer running in an AWS data center that you can use for pretty much anything—from hosting a website like MediaVault to running complex data analysis. You choose the operating system (Linux, Windows, etc.), the CPU, the memory, and the storage, giving you complete control over your computing resources.

---

### Choosing Your Engine: EC2 Instance Types

AWS offers a dizzying array of instance types, each optimized for different tasks. When the time comes, selecting the right one will be a balance between performance and cost. Here are the families I'll be considering:

* **T-family (Burstable):** These are great for applications with inconsistent traffic. The plan is to start MediaVault on a `t3.micro` or `t4g.small` instance to provide a baseline level of CPU performance with the ability to "burst" when needed, keeping initial costs low.
* **M-family (General Purpose):** If the application requires a more stable performance profile, these offer a good balance of compute, memory, and networking.
* **C-family (Compute Optimized):** While unlikely for the initial phase, these are for compute-intensive workloads that require high-performance processors.

---

### Handling Our Data: EC2 Storage Options 💾

An EC2 instance needs a place to store its operating system and files. AWS provides a few options, and choosing the right one is critical for performance, persistence, and cost.

* **Elastic Block Store (EBS):** Think of this as a network-attached virtual hard drive for your instance. EBS volumes are **persistent**, meaning the data remains intact even if you stop or restart your instance. They are the default and most common choice for boot volumes.
  * **General Purpose SSD (gp3):** This will be my go-to choice. It provides a great balance of price and performance for a wide variety of applications, including web servers.
  * **Provisioned IOPS SSD (io1/io2):** These are for high-performance databases or applications requiring very high I/O operations per second. Overkill for our initial needs, but good to know about.

* **Instance Store:** This is storage that is physically attached to the host computer running your instance. The main things to know are that it's **extremely fast** but also **ephemeral** (temporary). If the instance is stopped, terminated, or fails, all data on the instance store is **wiped forever**. This makes it great for temporary data like caches, buffers, or scratch data, but a terrible choice for anything you need to keep, like the operating system or user uploads.

For MediaVault, the plan is to use a **gp3 EBS volume** as the boot drive. This ensures our OS and application files are persistent and gives us a solid performance baseline without breaking the bank.

---

### The Bouncer: Planning Our Security Groups

A **Security Group** acts as a virtual firewall for an EC2 instance, controlling all inbound and outbound traffic. This is a non-negotiable security layer. You define rules that specify which protocols, ports, and IP addresses are allowed to connect.

My strategy will be to enforce the principle of least privilege from day one. I'll have to explicitly open only the ports required for MediaVault to function.

#### Classic Ports I'll Need to Configure

* **SSH (Port 22):** Essential for managing a Linux instance. When I launch the instance, I will configure the security group to allow SSH access *only* from my personal IP address.
* **HTTP (Port 80) & HTTPS (Port 443):** Once the web application is ready for deployment, I will update the security group to open these ports to the world (`0.0.0.0/0`) so users can access MediaVault.

---

### What's in a Name? An EC2 Naming Convention

As the project grows, a consistent naming convention will be a lifesaver. I plan to adopt a simple but effective structure:

`{project}-{environment}-{role}-{instance_number}`

For example: `mediavault-dev-webserver-01`

This name would immediately tell me that this is the first web server for the MediaVault project in the development environment.

---

### Estimating the Bill: How EC2 Pricing Works 💰

Before spending a single dollar, it's important to understand the costs. The **AWS Pricing Calculator** is the perfect tool for this. My cost estimate will be based on a few key factors:

1. **Instance Type:** A `t3.micro` will be significantly cheaper than an `m5.large`.
2. **Storage:** The size and type of the attached EBS volume will be part of the monthly cost.
3. **Region:** Costs vary slightly by AWS region.
4. **On-Demand vs. Savings Plans:** I'll start with On-Demand pricing for flexibility. As usage becomes predictable, I'll look into Savings Plans for significant discounts.
5. **Data Transfer:** Data transferred *out* of AWS to the internet incurs costs. This will be a key consideration for a media hosting application.

Having this blueprint ready is a huge step. When it's time to pull the trigger, I'll know exactly what to do. The next stage will be looking at database options to connect to our future instance.

---

### Follow Along

I'll be posting weekly updates on my progress, sharing code, and detailing my learnings. If you're interested in open-source development, data privacy, or learning AWS, I'd love for you to follow along.

Find me on **X (Twitter)** at [@amitalokbera_](https://twitter.com/amitalokbera_) for the latest updates and behind-the-scenes content. Let's build something cool together!
