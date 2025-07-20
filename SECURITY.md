# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of plot2llm seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### **Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**Email**: [orosero2405@gmail.com](mailto:orosero2405@gmail.com)

### What to include in your report

To help us understand and address the issue, please include:

1. **Description of the vulnerability**
   - What type of vulnerability is it?
   - What could an attacker do with this vulnerability?

2. **Steps to reproduce**
   - Provide clear, step-by-step instructions
   - Include code examples if applicable

3. **Impact assessment**
   - How severe is this vulnerability?
   - What systems or data could be affected?

4. **Suggested fix (optional)**
   - If you have ideas for how to fix the issue, please share them

5. **Your contact information**
   - We may need to contact you for additional information

### What happens next?

1. **Acknowledgment**: You will receive an acknowledgment within 48 hours
2. **Investigation**: We will investigate the reported vulnerability
3. **Updates**: We will keep you informed of our progress
4. **Resolution**: Once resolved, we will:
   - Release a security patch
   - Update the changelog
   - Credit you in the security advisory (if you wish)

### Responsible disclosure

We ask that you:

- **Do not disclose the vulnerability publicly** until we have had a chance to address it
- **Give us reasonable time** to respond and fix the issue
- **Work with us** to coordinate the disclosure

### Security best practices

When using plot2llm:

1. **Keep dependencies updated**: Regularly update plot2llm and its dependencies
2. **Use virtual environments**: Isolate your project dependencies
3. **Review code**: If using plot2llm in production, review the code you're running
4. **Monitor for updates**: Subscribe to GitHub releases for security updates

### Security features in plot2llm

plot2llm is designed with security in mind:

- **No network access**: plot2llm does not make any network requests
- **No file system access**: plot2llm only processes matplotlib/seaborn figures in memory
- **No code execution**: plot2llm does not execute any user-provided code
- **Input validation**: All inputs are validated before processing

### Contact

**Security Team**: [orosero2405@gmail.com](mailto:orosero2405@gmail.com)

**Project Maintainer**: [Osc2405](https://github.com/Osc2405)

---

Thank you for helping keep plot2llm secure! 🔒 