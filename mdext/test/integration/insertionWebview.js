// Build the actual mdext webview for Chromium UI tests without VS Code chrome.
const fs=require('node:fs'),path=require('node:path'),{pathToFileURL}=require('node:url');
const {getWebviewHtml}=require('../../src/webview');
const {renderMarkdown}=require('../../src/renderer');
(async()=>{
 const output=process.argv[2], markdown=fs.readFileSync(process.argv[3],'utf8');
 const context={extensionUri:{path:path.resolve(__dirname,'../..'),with:({path:p})=>pathToFileURL(p)}};
 let html=getWebviewHtml({cspSource:'file:',asWebviewUri:v=>v},context);
 html=html.replace('window.MathJax = {', 'window.outMessages=[];window.acquireVsCodeApi=()=>({postMessage:m=>{window.outMessages.push(m);return Promise.resolve();},getState:()=>({}),setState:()=>{}});window.MathJax = {');
 const result=await renderMarkdown({markdown,documentPath:process.argv[3],resolveAsset:(_,v)=>v,diagramRenderers:{}});
 fs.writeFileSync(output,html);
 fs.writeFileSync(output+'.json',JSON.stringify({...result,type:'render',documentKey:process.argv[3],title:'Insertion test'}));
})().catch(e=>{console.error(e);process.exit(1)});
