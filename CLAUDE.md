# Claude Embeddings Toolkit - Behavioral Instructions

## 🚨 CRITICAL: Intelligent Search Protocol

**YOU MUST** choose the right search strategy for each query. The goal is to find relevant text while avoiding context window bloat with irrelevant content.

### Required Workflow:
1. **ALWAYS activate Python environment FIRST**: `source code/virtual_env/bin/activate`
2. **Analyze the query type** before choosing search method
3. **Use semantic search (embeddings)** for concepts, themes, and broad topics
4. **Use lexical search (grep/sed)** for exact quotes, specific terms, and precise strings
5. **Use BOTH approaches** when appropriate - they complement each other
6. **Search multiple keywords in parallel** for comprehensive coverage

## 🔍 Choosing Between Semantic vs Lexical Search

### Use SEMANTIC SEARCH (Embeddings) when:
✅ **Exploring concepts or themes** ("What does this book say about marketing?")
✅ **Finding related ideas** ("Show me strategies for audience engagement")
✅ **Understanding context** ("How does the author approach sustainability?")
✅ **Cross-referencing topics** ("Connect climate change with food systems")
✅ **Discovering relevant sections** without knowing exact terminology

### Use LEXICAL SEARCH (Grep) when:
✅ **Finding exact quotes** ("Where does 'catastrophism has a long historical pedigree' appear?")
✅ **Locating specific terms** ("Find all mentions of 'vertical farming'")
✅ **Tracking precise phrases** ("Search for 'supply chain disruption'")
✅ **Following up on specific keywords** discovered during semantic search

### Use BOTH APPROACHES when:
✅ **Starting with semantic** to discover themes, then **lexical** for specific terms found
✅ **Starting with lexical** to find exact matches, then **semantic** for broader context
✅ **Comprehensive research** requiring both conceptual understanding and precise references

### Search Strategy Examples:

**Example 1: Concept Exploration**
- Query: "What does this book say about marketing strategies?"
- Approach: **Semantic first** → Find thematic sections about marketing
- Follow-up: **Lexical** → Search for specific strategies mentioned in semantic results

**Example 2: Quote Location**
- Query: "Where does the quote 'catastrophism has a long historical pedigree' appear?"
- Approach: **Lexical first** → `grep -r "catastrophism has a long historical pedigree"`
- Follow-up: **Semantic** → Search for related historical concepts for broader context

**Example 3: Comprehensive Analysis**
- Query: "Analyze the author's perspective on technological solutions"
- Approach: **Both in parallel** → Semantic for "technology solutions perspectives" + Lexical for "technolog*"

### Semantic Search (Embeddings) Commands:
```bash
# STEP 1: Always start here - broad search, no flags
python code/embeddings/search.py "[your query]"

# STEP 2: Only if too many results, narrow with:
python code/embeddings/search.py "[query]" --category [strategy/content/reference/planning]
python code/embeddings/search.py "[query]" --paths [directory/]

# STEP 3: Get full context when you find relevant chunks
python code/embeddings/retrieve.py --source [filename] --chunks [numbers]
```

### Lexical Search (Grep) Commands:
```bash
# Search for exact phrase across all files (excluding code/)
grep -r "exact phrase" --exclude-dir=code

# Case-insensitive search
grep -ri "phrase" --exclude-dir=code

# Show context around matches (3 lines before/after)
grep -r "phrase" -C 3 --exclude-dir=code

# Search specific file types
grep -r "phrase" --include="*.txt" --include="*.md"

# Combine with line numbers for citation
grep -rn "phrase" --exclude-dir=code
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
❌ **DON'T** use semantic search for exact quotes or specific strings
✅ **DO** use lexical search (grep) for precise text matching

❌ **DON'T** use lexical search alone for conceptual/thematic queries
✅ **DO** use semantic search (embeddings) for exploring ideas and themes

❌ **DON'T** start with narrow flags (`--category`, `--paths`) in semantic searches
✅ **DO** start broad, narrow only if too many results

❌ **DON'T** use single search terms or single approach
✅ **DO** use multiple parallel searches with different keywords and both methods when appropriate

❌ **DON'T** treat conversations as "new" without context
✅ **DO** search embeddings to understand project continuity

### Workflow Violations:
❌ **DON'T** choose wrong search type for the query (semantic for quotes, lexical for concepts)
❌ **DON'T** forget to activate Python environment before using embeddings
❌ **DON'T** skip diary entries for new learnings
❌ **DON'T** create files without checking existing resources first
❌ **DON'T** crowd context window with irrelevant results - be strategic about search method

## 🎯 Success Patterns

1. **Analyze query type first**: Determine if semantic, lexical, or both approaches are needed
2. **Research before suggesting**: Search appropriate systems before proposing solutions
3. **Document immediately**: Write diary entries when you learn something new
4. **Cross-reference intelligently**: Use multiple search approaches (semantic + lexical) for thorough coverage
5. **Retrieve full context**: Don't rely on search snippets alone - use retrieve.py for complete sections
6. **Maintain continuity**: Build on existing project knowledge, don't start fresh
7. **Optimize context usage**: Choose search methods strategically to avoid irrelevant results

---

**For technical setup, commands, and troubleshooting**: See README.md

**Remember**: Use semantic search (embeddings) for concepts and themes, lexical search (grep) for exact matches, and both for comprehensive research. The right tool for the right query keeps context clean and results relevant.