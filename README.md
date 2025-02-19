## Contributing

This sampling book runs statically, i.e. doesn't rerun the examples

```yaml
execute:
  execute_notebooks: 'off'
```

This is to reduce dependency management on the assumption that these are generally MWEs that are not strictly managed/checked.



Compose a jupyter notebook however one likes, and add it where appropriate to the file tree, along with an update to `_toc.yml` and `contributors.md`

if `jupyter-book` is installed locally, run the following to test the page builds

```bash
jb build .
```

Bibtex references can be added to the references.bib file and called in markdown by adding 
```markdown
{cite}`name of ref`
```


