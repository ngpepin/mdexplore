'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {applyInsert,cells}=require('../src/formInsertion');
const {hash,parseFields}=require('../src/formFields');
const req=(s,text='Hello')=>({operation:'insert',revision:hash(s),kind:'field',selection:{text,startLine:0,endLine:1,occurrence:0,occurrences:1}});
test('inserts after preserved selection with custom placeholder',()=>{
 const source='Hello world!\n',r={...req(source),label:'Name | "quoted" & café'};
 const next=applyInsert(source,r),f=parseFields(next)[0];
 assert.equal(f.placeholder,r.label);assert.equal(f.value,'');assert.equal(next.slice(0,f.start),'Hello ');assert.equal(next.slice(f.end),' world!\n');
});
test('checkbox defaults to Done with editable label',()=>{
 const source='Hello\n',r={...req(source),kind:'check'};
 assert.equal(parseFields(applyInsert(source,r))[0].label,'Done');
 r.label='Received | confirmed';assert.equal(parseFields(applyInsert(source,r))[0].label,r.label);
});
test('middle and final table columns preserve rows, alignment and escaped pipes',()=>{
 const source='| Left | Right |\r\n| :--- | ---: |\r\n| A \\| B | Target |\r\n| C | D |\r\n';
 for(const [column,text] of [[0,'A | B'],[1,'Target']]){
  const r={...req(source,text),newColumn:true};r.selection.endLine=4;r.selection.table={startLine:0,row:1,column};
  const next=applyInsert(source,r),lines=next.trimEnd().split('\r\n');
  assert.ok(lines.every(line=>cells(line).length===3));assert.ok(next.includes('A \\| B'));assert.ok(next.includes('Target'));assert.ok(next.includes('---:'));
  const [a,b]=cells(lines[2])[column+1];assert.match(lines[2].slice(a,b),/mdform:field/);assert.equal(parseFields(next).length,1);
 }
});
test('ambiguity, code, stale source and non-table column requests fail closed',()=>{
 let source='Hello Hello\n',r=req(source);assert.throws(()=>applyInsert(source,r),/unambiguously/);
 r.selection.occurrence=1;r.selection.occurrences=2;assert.ok(applyInsert(source,r).startsWith('Hello Hello <!--'));
 r.revision='stale';assert.throws(()=>applyInsert(source,r),/changed/);
 source='`Hello`\n';assert.throws(()=>applyInsert(source,req(source)),/outside code/);
 source='Hello\n';r={...req(source),newColumn:true};assert.throws(()=>applyInsert(source,r),/table cell/);
});
