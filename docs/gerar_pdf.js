// Gera o PDF da documentacao a partir do HTML.
// Roda de dentro de vitrine-bauru/web, que e onde o Playwright esta instalado:
//   node ../../baliza/docs/gerar_pdf.js ../../baliza/docs/P1-Baliza.html ../../baliza/docs/P1-Baliza.pdf
const path0 = require('path');
// O Playwright vive no node_modules de vitrine-bauru/web; resolvemos a partir do cwd.
const { chromium } = require(path0.join(process.cwd(), 'node_modules', '@playwright', 'test'));
const { pathToFileURL } = require('url');
const path = require('path');

(async () => {
  const entrada = path.resolve(process.argv[2]);
  const saida = path.resolve(process.argv[3]);
  const navegador = await chromium.launch();
  const p = await navegador.newPage();
  await p.goto(pathToFileURL(entrada).href, { waitUntil: 'networkidle' });
  await p.emulateMedia({ media: 'print' });
  await p.pdf({
    path: saida,
    format: 'A4',
    printBackground: true,
    margin: { top: '16mm', bottom: '18mm', left: '18mm', right: '18mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate:
      '<div style="width:100%;font-family:Segoe UI,Arial,sans-serif;font-size:8pt;color:#6b7a88;padding:0 18mm;display:flex;justify-content:space-between;">' +
      '<span>Baliza &middot; Projeto de Vis&atilde;o Computacional</span>' +
      '<span><span class="pageNumber"></span> de <span class="totalPages"></span></span></div>',
  });
  await navegador.close();
  console.log('PDF gerado em ' + saida);
})();
