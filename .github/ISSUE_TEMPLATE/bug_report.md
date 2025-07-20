---
name: Bug report
about: Create a report to help us improve plot2llm
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ['Osc2405']
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

A clear and concise description of what actually happened.

## Environment Information

**plot2llm version**: [e.g. 0.1.19]
**Python version**: [e.g. 3.9.7]
**Operating System**: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
**matplotlib version**: [e.g. 3.5.0]
**seaborn version**: [e.g. 0.11.0]

## Code Example

```python
import matplotlib.pyplot as plt
import plot2llm

# Your code here
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
result = plot2llm.convert(fig, 'text')
print(result)
```

## Error Message

```
# Paste the full error message here
```

## Additional Context

Add any other context about the problem here, such as:
- Screenshots if applicable
- Related issues
- Workarounds you've tried

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included all relevant environment information
- [ ] I have tested with the latest version of plot2llm 