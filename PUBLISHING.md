# Publishing to GitHub

These instructions are for the maintainer. Students only need the README.

## Create the empty remote

On GitHub's **Create a new repository** screen:

- Owner: `ubcbraincircuits`
- Repository name: `placefield_lab`
- Suggested description: `A teaching model of mouse CA1 place cells, elevated-plus-maze exploration, and neural position decoding.`
- Choose the visibility you intend for the class. Use Public if anyone should be able to browse and clone it.
- Leave **Add a README**, **Add .gitignore**, and **Choose a license** unchecked/unselected.
  The local project supplies all three, including the MIT license.

Then click **Create repository**. Starting with an empty remote avoids a second,
unrelated initial commit. This follows [GitHub's existing-code import guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## Push the prepared local repository

Open a terminal **inside the `placefield_lab` project directory**, where `run.py`
and `README.md` live. Use Git's existing authentication setup; HTTPS may open a
GitHub sign-in flow. Do not put a password or token into the remote URL.

If the folder has not yet been initialized and committed:

```bash
git init -b main
git add .
git status --short
git commit -m "Add synthetic CA1 place-field teaching lab"
```

Inspect `git status` before committing. The intended repository includes source,
documentation, tests, the MIT license, and the curated gallery. It excludes virtual
environments and regenerated `results/` or `experiments/` folders.

For a local repository that is already committed, connect and push:

```bash
git remote -v
# Run the next line only if no origin remote exists:
git remote add origin https://github.com/ubcbraincircuits/placefield_lab.git
git push -u origin main
```

If `origin` already exists, verify its URL instead of adding it again. If GitHub
already contains commits, fetch and inspect that history before combining it with
the local repository; do not use a force push to bypass the difference.

## Verify the published result

1. Open the README on GitHub and check that the GIF and image links display.
2. Open the plot interpretation guide and confirm its linked figures display.
3. Check the **Actions** tab for model checks and rendering on the three operating systems.
   A workflow file being present does not mean those remote checks have passed yet.
4. Try the README quick start from a fresh clone or ZIP extraction.

The full synthetic dataset can be regenerated with `run.py`. If you later want to
share a frozen dataset, attach an archive to a GitHub Release rather than committing
every repeated experiment into the source history.

## If you prefer a graphical Git client

In GitHub Desktop, add the existing local repository and verify that its remote is
the intended organization/repository. Use **Push origin** after connecting it to the
empty remote. The Git command above remains the shortest route when Git is installed.
