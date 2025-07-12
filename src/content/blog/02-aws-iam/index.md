---
title: "Mastering AWS IAM"
description: "MediaVault Deep Dive: Mastering AWS IAM for Secure Access"
date: "Jul 12 2025"
---

Welcome back to my series on building MediaVault! This week, I've been wrestling with a foundational piece of any AWS project: **Identity and Access Management (IAM)**. Getting IAM right is crucial for security, so I want to share what I've learned about its core components: Users, Groups, and Policies.

---

### What is AWS IAM? 🤔

At its heart, IAM is the service that lets you control who can do what within your AWS account. It's the gatekeeper of your cloud resources. Instead of sharing your main account credentials (a huge security no-no!), you create IAM entities to grant specific permissions to people and applications.

The three pillars of IAM are:

1. **Users:** An IAM User is an entity you create in AWS to represent the person or application that interacts with your AWS resources. Each user has its own unique security credentials (a password for the console and access keys for the API/CLI).

2. **Groups:** A Group is simply a collection of IAM users. They are a powerful way to manage permissions for multiple users at once. Instead of attaching permissions to each individual user, you can assign them to a group. If a user is in that group, they automatically inherit the group's permissions. This makes managing access much easier as your team grows. For example, you could have a `Developers` group and an `Admins` group, each with different levels of access.

3. **Policies:** This is where the real power lies. A Policy is a JSON document that explicitly defines permissions. It states what actions are allowed or denied on which AWS resources. For example, a policy could grant a user read-only access to a specific S3 bucket or allow them to start and stop EC2 instances. Policies can be attached directly to users or, more commonly, to groups.

---

### Users, Groups, and Policies: A Visual Guide

To make this clearer, the diagram below breaks down AWS IAM into two core concepts: Identities and Permissions. Identities represent who is making the request—such as users, groups, roles, or credentials. Permissions represent what those identities are allowed to do, and are managed through policies made up of permission statements.
This visual structure helps illustrate that IAM policies can be attached to both users and groups, and users can inherit permissions via group membership. Roles, on the other hand, enable temporary access for trusted entities or services.

![IAM Users, Groups, and Policies](./iam.png)
<sub>Credits: [K12Academy](https://k21academy.com)</sub>

---

### Securing Your Account: Multi-Factor Authentication (MFA) 🔐

One of the most critical security measures you can implement is **Multi-Factor Authentication (MFA)**. MFA adds an extra layer of protection on top of your username and password. With MFA enabled, when a user signs in, they must provide their password AND a unique authentication code from a physical or virtual MFA device (like Google Authenticator or Authy on your phone).

For your root AWS account user, enabling MFA is non-negotiable. For all other IAM users, it's a strongly recommended best practice.

---

### IAM Best Practices ✅

As I've been setting up IAM for MediaVault, I've been following these best practices, and you should too:

* **Lock away your root user access keys:** Don't use your root account for everyday tasks. Enable MFA on it and store the credentials securely.
* **Create individual IAM users:** Don't share credentials. Create a separate user for each person or application that needs access.
* **Use groups to assign permissions:** Manage permissions at scale by organizing users into groups that align with their job functions.
* **Grant least privilege:** This is the golden rule. Only grant the minimum permissions necessary for a user or service to perform its tasks. Start with a minimal set of permissions and grant additional permissions as needed.
* **Enable and enforce MFA:** As mentioned, this is a simple yet powerful way to enhance security.
* **Rotate credentials regularly:** Change passwords and access keys periodically to reduce the risk of compromised credentials.
* **Use roles for applications:** For applications running on EC2 instances, use IAM Roles to provide temporary credentials instead of storing long-term access keys on the instance.

That’s a wrap for this week! Getting a solid IAM foundation in place is a huge step forward for MediaVault. Next blog, I'll be diving into setting up our first database.

---

### Follow Along

I'll be posting weekly updates on my progress, sharing code, and detailing my learnings. If you're interested in open-source development, data privacy, or learning AWS, I'd love for you to follow along.

Find me on **X (Twitter)** at [@amitalokbera_](https://twitter.com/amitalokbera_) for the latest updates and behind-the-scenes content. Let's build something cool together!
