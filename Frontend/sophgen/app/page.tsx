"use client";

import React, { useEffect, useRef, useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { EditorView, Decoration } from "@codemirror/view";
import { StateEffect } from "@codemirror/state";
import { cpp } from "@codemirror/lang-cpp";
import ReactMarkdown from "react-markdown";
import { AiOutlineLoading } from "react-icons/ai";


export default function Home() {
  const [code, setCode] = useState("// Enter C code here");
  const [outputCode, setOutput] = useState("// Results will appear here");
  const [result, setResult] = useState(
    "Paste the c code and press generate to start the process"
  );
  const [instructions, setInstructions] = useState<string[]>([]);
  const [currentInstruction, setCurrentInstruction] = useState<string>("");
  const [evaluationResult, setEvaluationResult] = useState("");
  const [highlightedLines, setHighlightedLines] = useState<number[]>([]);
  const [isLoading, setLoading] = useState(false);
  const [huntingMode, setHuntingMode] = React.useState(false);

  let currentResult = "";

  const resultEditorRef = useRef<EditorView | null>(null);

  // Handle adding a new instruction
  const handleAddInstruction = () => {
    if (currentInstruction.trim()) {
      setInstructions([...instructions, currentInstruction]); // Add the instruction to the array
      setCurrentInstruction(""); // Clear the input after adding
    }
  };

  // useEffect to detect changes in the instructions array
  useEffect(() => {
    if (instructions.length > 0) {
      // Get the last instruction
      const lastInstruction = instructions[instructions.length - 1];

      // Split the instruction and line number
      const [instruction, lineNumberStr] = lastInstruction.split("!@#$");

      const cleanedLineNumberStr = lineNumberStr.replace(/[^\d:]/g, "").trim();
      // Parse the line number (e.g., "5:6" -> [5, 6])
      const [startLine, endLine] = cleanedLineNumberStr.split(":").map(Number);

      // Highlight each line from startLine to endLine (inclusive)
      for (let i = startLine; i <= endLine; i++) {
        setHighlightedLines((prevLines) =>
          prevLines.includes(i) ? prevLines : [...prevLines, i]
        );
      }
    }
  }, [instructions]); // This effect will run when the instructions array changes

  function cleanSourceCode(code: string): string {
    // Remove single-line comments (//)
    let cleanedCode = code.replace(/\/\/.*$/gm, "");

    // Remove multi-line comments (/* */)
    cleanedCode = cleanedCode.replace(/\/\*[\s\S]*?\*\//g, "");

    // Remove extra whitespace and empty lines
    cleanedCode = cleanedCode.replace(/\n\s*\n+/g, "\n"); // Remove multiple newlines
    cleanedCode = cleanedCode.trim(); // Remove leading and trailing whitespace

    return cleanedCode;
  }

  const handleEvaluate = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/process_llm/", {
        method: "POST", // Specify the request method
        headers: {
          "Content-Type": "application/json", // Set the content type to JSON
        },
        body: JSON.stringify({
          name: "processed_code", // Static name
          operations: instructions, // The instructions array you want to send
        }),
      });

      // Check if the response is successful (status 200-299)
      if (response.ok) {
        const data = await response.json(); // Parse the JSON response
        setEvaluationResult(data.response);
        if (highlightedLines.length > 0) {
          // Function to highlight lines with a timeout of 200ms
          highlightedLines.forEach((lineNumber, index) => {
            setTimeout(() => {
              // Highlight the line here (you can add your highlight logic)
              highlightLine(lineNumber);
            }, 200 * (index + 1)); // Delay each line highlight by 200ms
          });
        }
      } else {
        setEvaluationResult("Failed to get evaluation results.");
      }
    } catch (error) {
      console.error("Error during request:", error);
    }
  };

  const handleGenerate = async () => {
    try {
      const cleaned_code = cleanSourceCode(code);
      setLoading(true);
      // Create a FormData object to send the form data
      const formData = new FormData();
      formData.append("code", cleaned_code); // Replace with the actual code from the editor or input field

      // Send the form data to the server
      const locResponse = await fetch("http://127.0.0.1:8000/process_code/", {
        method: "POST",
        body: formData, // Send the form data as the body
      });

      console.log(locResponse);

      // Check if the response is successful
      if (!locResponse.ok) {
        throw new Error("Failed to fetch data");
      }

      // Parse the JSON response
      const loc_data = await locResponse.json();

      // Ensure predictions is an array
      if (Array.isArray(loc_data.predictions)) {
        // Format predictions with indices
        const loc_predictions = loc_data.predictions
          .map(
            (prediction: string, index: number) =>
              ` [${index + 1}]. ${prediction}`
          )
          .join("\n\n");

        currentResult =
          "**Running Localization Prediction...**\n\n" + loc_predictions;
        // Set the result to include the predictions with their indices
        setResult(currentResult);

        // Step 2
        const astResponse = await fetch(
          "http://127.0.0.1:8000/process_ast/?name=processed_code"
        );

        // Check if the GET request was successful
        if (!astResponse.ok) {
          throw new Error("Failed to fetch AST");
        }
        // Parse the GET response (JSON)
        const ast_data = await astResponse.json();
        // Ensure ast_data.patterns is an array or string before using .length
        const patternsLength =
          Array.isArray(ast_data.patterns) ||
          typeof ast_data.patterns === "string"
            ? ast_data.patterns.length
            : 0;

        currentResult +=
          "\n\n**Running Pattern Extraction...**\n\n" +
          `${patternsLength} patterns found.`;
        setResult(currentResult);

        //Step 3
        const patternResponse = await fetch(
          "http://127.0.0.1:8000/process_pattern/?name=processed_code"
        );

        if (!patternResponse.ok) {
          throw new Error("Failed to fetch Pattern");
        }
        const pattern_data = await patternResponse.json();

        // Access the 'output' array
        const output = pattern_data.output;

        setOutput(code);
        setInstructions(output);

        // Iterate over each element in the 'output' array
        const parsedPatterns = output.map((line: string) => {
          // Split each line by the separator '!@#$'
          const [instruction, lineNumber] = line.split("!@#$");

          return {
            instruction: instruction.trim(), // Remove any extra spaces
            lineNumber: lineNumber.trim(), // Remove any extra spaces
          };
        });
        currentResult += "\n\n**Running Instruction Extraction...**\n\n";

        // Iterate over the parsed patterns and update currentResult
        parsedPatterns.forEach((pattern: any) => {
          currentResult += `++ Extracted Instruction:\n\`\`\`${pattern.instruction}\`\`\` at line ${pattern.lineNumber}\n`;
        });

        setResult(currentResult);

        if (highlightedLines.length > 0) {
          // Function to highlight lines with a timeout of 200ms
          highlightedLines.forEach((lineNumber, index) => {
            setTimeout(() => {
              // Highlight the line here (you can add your highlight logic)
              highlightLine(lineNumber);
            }, 200 * (index + 1)); // Delay each line highlight by 200ms
          });
        }
        setLoading(false);
      } else {
        console.error("Predictions is not an array", loc_data.predictions);
        setLoading(false);
      }
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
    }
  };

  const handleClear = () => {
    setCode("// Enter C code here"); // Clear the code editor
    setOutput("// Results will appear here"); // Clear the result
    setResult("Paste the c code and press generate to start the process");
    setEvaluationResult("Run with a list of operations to start evaluating");
    setInstructions([]);
    setHighlightedLines([]);
  };

  const highlightLine = (lineNumber: number) => {
    if (!resultEditorRef.current) return;

    const editor = resultEditorRef.current;
    const totalLines = editor.state.doc.lines;

    // Ensure the line number is within range
    if (lineNumber < 1 || lineNumber > totalLines) {
      console.error(
        `Invalid line number ${lineNumber}. The document has ${totalLines} lines.`
      );
      return;
    }

    const line = editor.state.doc.line(lineNumber);

    const highlightDecoration = Decoration.line({
      attributes: {
        class: "bg-red-400 cursor-pointer",
        "data-line": `${lineNumber}`, // Add a data-line attribute
      },
    });

    const decorations = Decoration.set([highlightDecoration.range(line.from)]);

    const highlightExtension = EditorView.decorations.of(() => decorations);

    editor.dispatch({
      effects: StateEffect.appendConfig.of(highlightExtension),
    });
  };

  return (
    <div className="min-h-screen w-full bg-white">
      <header className="w-full p-4 border-b border-gray-200">
        <h1 className="text-2xl font-bold text-center text-slate-700">
          SOPHGEN
        </h1>
      </header>

      <main className="p-6 max-w-6xl mx-auto">
        <h2 className="text-xl text-slate-600 text-center mb-8">
          Welcome to SOPHGEN! A Powerful Tool for Generating Synthetic
          Vulnerabilities
        </h2>

        <div className="flex justify-center gap-2 mb-2">
          <button
            className={`px-4 text-sm py-2 rounded w-[100px] flex items-center justify-center ${
              isLoading
                ? "bg-orange-300 cursor-not-allowed"
                : "bg-orange-500 hover:bg-orange-600"
            }`}
            onClick={handleGenerate}
            disabled={isLoading}
          >
           Generate
          </button>

          <button
            className="px-4 text-sm py-2 bg-gray-500 text-white rounded hover:bg-gray-600 w-[100px]"
            onClick={handleClear}
            disabled={isLoading}
          >
            Clear
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="bg-white p-4 rounded-lg shadow-lg border border-gray-200">
            <h3 className="text-sm font-medium mb-2 text-slate-600">
              Input Code
            </h3>
            <CodeMirror
              value={code}
              height="256px"
              extensions={[cpp()]}
              onChange={setCode}
              theme="light"
              className="text-slate-600"
              style={{ border: "1px solid #e5e7eb" }}
            />
          </div>

          <div className="bg-white p-4 rounded-lg shadow-lg border border-gray-200">
            <h3 className="text-sm font-medium mb-2 text-slate-600">Results</h3>
            <CodeMirror
              value={outputCode}
              height="256px"
              extensions={[cpp()]}
              editable={false}
              theme="light"
              className="text-slate-600"
              style={{ border: "1px solid #e5e7eb" }}
              onCreateEditor={(editor) => {
                resultEditorRef.current = editor;
              }}
            />
          </div>
        </div>

        <div className="border-2 shadow-md border-gray-200 p-4 bg-white w-full mb-3">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium text-slate-600">Evaluation</h3>
          </div>
          <div className="w-full border-dashed border-2 p-2 bg-gray-100">
            <ReactMarkdown className="text-sm text-slate-600">
              {evaluationResult}
            </ReactMarkdown>
          </div>
          <div className="flex justify-end mt-4 gap-2">
            <button
              className="px-4 text-sm py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={handleEvaluate}
            >
              Get Evaluation
            </button>
          </div>
        </div>

        <div className="flex mb-6 flex-row w-full gap-3">
          {/* Left Section (Explanation and Instruction Input) */}
          <div className="border-2 shadow-md border-gray-200 p-4 bg-white w-3/4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-medium text-slate-600">Process</h3>
              <h3 className="text-sm font-medium text-slate-600 text-right">
                Time taken: 53.44s
              </h3>
            </div>
            <div className="w-full border-dashed border-2 p-2 bg-gray-100">
              <ReactMarkdown className="text-sm text-slate-600">
                {result}
              </ReactMarkdown>
            </div>
            <div className="flex justify-end mt-4 gap-2">
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  checked={huntingMode}
                  onChange={(e) => setHuntingMode(e.target.checked)}
                  className="w-4 h-4"
                />
                Hunting Mode
              </label>
              <input
                type="text"
                value={currentInstruction}
                onChange={(e) => setCurrentInstruction(e.target.value)}
                className="border p-2 rounded text-sm text-slate-600"
              />
              <button
                className="px-4 py-2 bg-green-500 text-sm text-white rounded hover:bg-green-600"
                onClick={handleAddInstruction}
              >
                Add Instruction
              </button>
            </div>
          </div>

          {/* Right Section (Current Instructions List) */}
          <div className="text-sm text-slate-600 border-dotted border-2 border-gray-300 p-2 w-1/4">
            <h3>Current Instructions:</h3>
            <ul>
              {instructions.map((instruction, index) => (
                <li className={"text-purple-500"} key={index}>
                  {index}: {instruction}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
