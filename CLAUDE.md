# Claude Embeddings Toolkit - Behavioral Instructions

## 🚨 CRITICAL: Mandatory Usage Protocol

**YOU MUST** use the embeddings system for ALL content research and file discovery. This is mandatory for proper project context.

### Required Workflow - NO EXCEPTIONS:
1. **ALWAYS activate Python environment FIRST**: `source code/virtual_env/bin/activate`
2. **MANDATORY: Use embeddings for ALL files outside of code/**
3. **Start broad, narrow only if needed**: `python code/embeddings/search.py "[query]"`
4. **Never bypass embeddings** for content research - this causes context loss
5. **Search multiple keywords in parallel** for comprehensive coverage

### When You MUST Use Embeddings:
✅ **Any file research** (finding documents, references, strategies)  
✅ **Content analysis** (understanding existing materials)  
✅ **Project context** (learning about user preferences/history)  
✅ **Cross-referencing** (connecting related concepts)  
✅ **Before making recommendations** (check existing resources first)

### Embeddings Search Priority:
```bash
# STEP 1: Always start here - broad search, no flags
python code/embeddings/search.py "[your query]"

# STEP 2: Only if too many results, narrow with:
python code/embeddings/search.py "[query]" --category [strategy/content/reference/planning]
python code/embeddings/search.py "[query]" --paths [directory/]

# STEP 3: Get full context when you find relevant chunks
python code/embeddings/retrieve.py --source [filename] --chunks [numbers]
```

## 🎭 Diary System Protocol

**YOU MUST** write diary entries when you learn something new about the project or user.

### Auto-Created Structure:
```
diaries/
├── claude/      # Claude's insights, decisions, learnings
└── user/        # User's process, preferences, reflections
```

### Required Diary Entry Format:
```markdown
# Learning: [Brief Title]

**Source**: [Direct conversation/research/discovery]
**Relevance**: [How this impacts your project]

## What I Learned
[Detailed documentation of the insight]

## Strategic Impact  
[How this affects decisions/workflow]
```

### When to Write Diary Entries:
- User preferences or behavior patterns discovered
- Project insights from conversation (not file analysis)
- Strategic decisions or direction changes
- Technical discoveries about tools/systems
- Key relationships or contacts mentioned
- Creative process observations

## 📋 Common Mistake Prevention

### Search Strategy Errors:
❌ **DON'T** start with narrow flags (`--category`, `--paths`)  
✅ **DO** start broad, narrow only if too many results

❌ **DON'T** use single search terms  
✅ **DO** use multiple parallel searches with different keywords

❌ **DON'T** treat conversations as "new" without context  
✅ **DO** search embeddings to understand project continuity

### Workflow Violations:
❌ **DON'T** bypass embeddings for "quick" file research  
❌ **DON'T** forget to activate Python environment  
❌ **DON'T** skip diary entries for new learnings  
❌ **DON'T** create files without checking existing resources first

## 🎯 Success Patterns

1. **Research first**: Always search embeddings before suggesting solutions
2. **Document immediately**: Write diary entries when you learn something new
3. **Cross-reference everything**: Use multiple search approaches for thorough coverage
4. **Retrieve full context**: Don't rely on search snippets alone
5. **Maintain continuity**: Build on existing project knowledge, don't start fresh

---

**For technical setup, commands, and troubleshooting**: See README.md

**Remember**: The embeddings system is your project memory. Use it religiously for proper context and continuity.