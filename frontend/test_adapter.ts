import { adaptAnalyzeResponse } from './src/lib/adapter';
import * as fs from 'fs';

const data = JSON.parse(fs.readFileSync('../test.json', 'utf-8'));
const result = adaptAnalyzeResponse(data);
console.log(JSON.stringify(result, null, 2));
