# IVC Final Submission Checklist

## Status Summary
- `PASS` Main manuscript compiled: `manuscript_ivc.pdf` (`18` pages)
- `PASS` Supplement compiled: `supplement_ivc.pdf` (`6` pages)
- `PASS` Cover letter compiled: `cover_letter_ivc.pdf`
- `PASS` Highlights file present with `4` bullets, each within the `<=85` character limit
- `PASS` Graphical abstract present as `PNG` and `PDF`
- `PASS` Graphical abstract canvas adjusted to the IVC-style landscape ratio (`2656x1062`)
- `PASS` Final package assembled: `submission_package_final/`
- `PASS` Zip package assembled: `submission_package_final.zip`
- `PASS` Main manuscript uses `elsarticle.cls` and `elsarticle-num.bst`
- `PASS` Main manuscript includes AI-use disclosure before the references

## Files To Upload
- Manuscript source bundle:
  - `manuscript.tex`
  - `references.bib`
  - `elsarticle.cls`
  - `elsarticle-num.bst`
  - `figures/*`
  - `tables/*`
- Manuscript PDF:
  - `manuscript.pdf`
- Supplementary material:
  - `supplement.pdf`
- Cover letter:
  - `cover_letter.pdf`
- Highlights:
  - `highlights.txt`
- Graphical abstract:
  - `graphical_abstract.png`

## Submission Portal
- Primary live entry:
  - `https://submit.elsevier.com/IMAVIS`
- Legacy landing page:
  - `https://www.editorialmanager.com/IMAVIS`
- If the legacy page shows "under development", use the new Elsevier submission route above.

## Manual Portal Actions Still Required
- Fill all author metadata in the portal.
- Fill the corresponding author contact details in the portal.
- Upload or complete the declarations form requested by Elsevier's submission system.
- Choose the subscription / non-OA route in the portal.
- Enter suggested reviewers manually if the form asks for them.
  - Internal helper only: `reviewer_shortlist.md`

## Final Manual Checks Before Clicking Submit
- Confirm the title exactly matches the manuscript PDF.
- Confirm the abstract pasted into the portal matches the PDF.
- Confirm the keyword list matches the PDF.
- Confirm all author names, order, and emails are correct.
- Confirm the affiliation wording you want to expose in the portal.
  - Current manuscript wording: `Faculty of Automation, Guangdong University of Technology, Guangzhou, China`
  - If you want to add a postal code such as `510006`, verify it manually before editing the manuscript or portal metadata.
- Confirm the graphical abstract preview is not cropped in the portal.
- Confirm the supplement is labeled as supplementary material, not as a revised manuscript.
- Do not upload `reviewer_shortlist.md` unless the system explicitly asks for a reviewer file.

## Recommended Upload Order
1. `manuscript.pdf`
2. `manuscript.tex`, `references.bib`, `elsarticle.cls`, `elsarticle-num.bst`
3. `figures/*`
4. `tables/*`
5. `highlights.txt`
6. `graphical_abstract.png`
7. `cover_letter.pdf`
8. `supplement.pdf`

## Final Package Paths
- Folder:
  - `submission/ivc_final/`
- Original local source before curation:
  - local build folder, now represented by `submission/ivc_final/`
- Zip:
  - The original local zip was unpacked before committing.
  - The GitHub release keeps the extracted package to avoid duplicate storage.

