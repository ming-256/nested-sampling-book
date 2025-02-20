## Contributing

This sampling book runs statically, i.e. doesn't rerun the examples
```yaml
execute:
  execute_notebooks: 'off'
```

This is to reduce dependency management on the assumption that these are generally MWEs that are not strictly managed/checked.


## Contributing guide

- Make an example notebook and run it locally. The visual state of the notebook is copied to the book page.
  - Include references to codes / non-standard dependencies as links (see the quickstart example)
  - Include bibtex references in the references.bib file and call them in markdown by adding `{cite}`name of ref``
- Add the notebook to the appropriate part of the file tree and track it in git
- Edit the `_toc.yml` file to include the notebook in the table of contents (and hence have it picked up by jupyter book)
- Add yourself to the contributors.md file
- `pip install jupyter-book` to install locally then, run the following to test the page builds 
```bash
jb build .
```



