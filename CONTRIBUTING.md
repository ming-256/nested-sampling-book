# Contributions

Desirable contributions for this repository are primarily short notebooks that demonstrate the use of the nested sampling algorithm on physics problems. Typically this will involve some modelling code / likelihood code that is written in jax, and would like to capitalise on the hardware acceleration possible whilst still computing partition functions/model evidences.

## Contributing guide

- Raise an issue or submit a PR to discuss the example you would like to add.
- Clone the repo and switch to a new branch to contribute your example via a PR.
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