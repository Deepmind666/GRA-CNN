<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <defs>
    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1f77b4" />
    </marker>
    <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2ca02c" />
    </marker>
     <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d62728" />
    </marker>
    <linearGradient id="cnn-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#e1f5fe;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#b3e5fc;stop-opacity:1" />
    </linearGradient>
    <style type="text/css">
      .label-text { font-family: Arial, sans-serif; font-size: 16px; fill: #333; text-anchor: middle; }
      .title-text { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #222; text-anchor: middle; }
      .formula-text { font-family: 'Times New Roman', serif; font-size: 20px; fill: #000; text-anchor: middle; font-style: italic;}
      .flow-line { fill: none; stroke-width: 4; }
      .box-bg { fill: #f9f9f9; stroke: #ccc; stroke-width: 2; rx: 8; ry: 8; }
      .highlight-box { fill: #fff3e0; stroke: #ffb74d; stroke-width: 3; rx: 12; ry: 12; }
    </style>
  </defs>

  <rect width="100%" height="100%" fill="#ffffff" />

  <g transform="translate(500, 30)">
    <rect x="-60" y="0" width="120" height="80" fill="#ddd" stroke="#999" rx="5"/>
    <path d="M-40 60 L-20 30 L0 50 L20 20 L40 60 Z" fill="#bbb"/>
    <circle cx="25" cy="25" r="10" fill="#bbb"/>
    <text x="80" y="45" class="title-text" text-anchor="start">Input Image</text>
    
    <line x1="0" y1="85" x2="0" y2="135" class="flow-line" stroke="#1f77b4" marker-end="url(#arrow-blue)"/>
  </g>

  <g transform="translate(400, 140)">
    <rect x="0" y="0" width="400" height="100" fill="url(#cnn-grad)" stroke="#1f77b4" rx="8"/>
    <text x="200" y="30" class="title-text">Pre-Trained CNN</text>
    <g transform="translate(30, 50)">
        <rect x="0" y="0" width="40" height="40" fill="#90caf9" stroke="#1f77b4"/> <text x="20" y="25" font-size="10" text-anchor="middle">Conv1</text>
        <rect x="50" y="10" width="30" height="30" fill="#81d4fa" stroke="#1f77b4"/> <text x="65" y="30" font-size="10" text-anchor="middle">Pool1</text>
        <rect x="90" y="0" width="40" height="40" fill="#90caf9" stroke="#1f77b4"/> <text x="110" y="25" font-size="10" text-anchor="middle">Conv2</text>
        <rect x="140" y="10" width="30" height="30" fill="#81d4fa" stroke="#1f77b4"/> <text x="155" y="30" font-size="10" text-anchor="middle">Pool2</text>
        <line x1="180" y1="25" x2="200" y2="25" stroke="#1f77b4" stroke-width="2"/>
        <rect x="200" y="5" width="20" height="40" fill="#b39ddb" stroke="#5e35b1"/> <text x="210" y="28" font-size="10" text-anchor="middle" transform="rotate(-90, 210, 28)">FC1</text>
        <rect x="230" y="5" width="20" height="40" fill="#b39ddb" stroke="#5e35b1"/> <text x="240" y="28" font-size="10" text-anchor="middle" transform="rotate(-90, 240, 28)">FC2</text>
    </g>
  </g>


  <path d="M500 245 L500 280 L300 280 L300 315" class="flow-line" stroke="#2ca02c" marker-end="url(#arrow-green)"/>
  <path d="M700 245 L700 280 L900 280 L900 315" class="flow-line" stroke="#2ca02c" marker-end="url(#arrow-green)"/>

  <g transform="translate(150, 320)">
    <rect x="0" y="0" width="300" height="120" class="box-bg"/>
    <text x="150" y="30" class="title-text" fill="#1f77b4">Feature Maps &amp; Activations (a<tspan dy="5" font-size="12">c</tspan>)</text>
    <path d="M30 70 C 60 30, 90 110, 120 70 S 180 30, 210 70 S 270 110, 270 70" fill="none" stroke="#1f77b4" stroke-width="3"/>
    <path d="M30 70 C 50 50, 70 90, 90 70 S 130 50, 150 70" fill="none" stroke="#4fc3f7" stroke-width="2" opacity="0.6"/>
  </g>

  <g transform="translate(750, 320)">
    <rect x="0" y="0" width="300" height="120" class="box-bg"/>
    <text x="150" y="30" class="title-text" fill="#2ca02c">Logits (z)</text>
    <text x="150" y="100" class="label-text">z = Softmax(W<tspan dy="5" font-size="12">f</tspan>*a<tspan dy="0" font-size="12">f</tspan>)</text>
    <path d="M30 70 C 60 20, 90 120, 120 70 S 180 20, 210 70 S 270 120, 270 70" fill="none" stroke="#2ca02c" stroke-width="3"/>
    <path d="M30 70 C 50 90, 70 50, 90 70 S 130 90, 150 70" fill="none" stroke="#81c784" stroke-width="2" opacity="0.6"/>
  </g>

  <line x1="300" y1="445" x2="435" y2="445" class="flow-line" stroke="#2ca02c" marker-end="url(#arrow-green)"/>
  <line x1="900" y1="445" x2="765" y2="445" class="flow-line" stroke="#2ca02c" marker-end="url(#arrow-green)"/>

  <g transform="translate(450, 360)">
    <rect x="0" y="0" width="300" height="170" class="highlight-box"/>
    <rect x="0" y="0" width="300" height="40" fill="#ffb74d" rx="12" ry="12" style="clip-path: inset(0 0 70% 0);"/>
    <text x="150" y="28" class="title-text" fill="#fff">GRA Calculation</text>
    <text x="150" y="55" class="label-text" font-size="14">(Semantic Alignment)</text>
    
    <g transform="translate(150, 100)">
      <text x="0" y="-10" class="formula-text">ξ<tspan dy="5" font-size="14">c</tspan>(i) = <tspan font-family="Arial" font-style="normal">∂</tspan>L / <tspan font-family="Arial" font-style="normal">∂</tspan>a<tspan dy="5" font-size="14">c</tspan>(i) · a<tspan dy="0" font-size="14">c</tspan>(i)</text>
      <text x="0" y="35" class="formula-text">γ<tspan dy="5" font-size="14">c</tspan> = E[ξ<tspan dy="5" font-size="14">c</tspan>(i)]</text>
    </g>
  </g>

  <line x1="600" y1="535" x2="600" y2="575" class="flow-line" stroke="#1f77b4" marker-end="url(#arrow-blue)"/>

  <g transform="translate(450, 580)">
    <rect x="0" y="0" width="300" height="120" class="box-bg"/>
    <text x="150" y="25" class="title-text">Channel Ranking by γ<tspan dy="5" font-size="12">c</tspan></text>
    <g transform="translate(40, 100) scale(1, -1)">
      <rect x="0" y="0" width="20" height="70" fill="#1f77b4"/>
      <rect x="30" y="0" width="20" height="60" fill="#2196f3"/>
      <rect x="60" y="0" width="20" height="45" fill="#42a5f5"/>
      <rect x="90" y="0" width="20" height="30" fill="#64b5f6"/>
      <rect x="120" y="0" width="20" height="20" fill="#90caf9"/>
      <rect x="150" y="0" width="20" height="15" fill="#bbdefb"/>
      <rect x="180" y="0" width="20" height="10" fill="#e3f2fd" stroke="#d62728" stroke-width="2"/> </g>
    <text x="40" y="45" class="label-text" font-size="12">High γ<tspan dy="3" font-size="10">c</tspan></text>
    <text x="220" y="105" class="label-text" font-size="12">Low γ<tspan dy="3" font-size="10">c</tspan></text>
    <text x="130" y="45" font-size="12">γ<tspan dy="3" font-size="10">1</tspan> > γ<tspan dy="0" font-size="10">2</tspan> > ... > γ<tspan dy="0" font-size="10">n</tspan></text>
  </g>


  <path d="M445 650 L350 650" class="flow-line" stroke="#d62728" stroke-dasharray="8,4" marker-end="url(#arrow-red)"/>

  <g transform="translate(150, 590)">
    <text x="100" y="20" class="title-text" fill="#d62728">Prune Low-Score</text>
    <text x="100" y="45" class="title-text" fill="#d62728">Channels</text>
    
    <g transform="translate(50, 60)">
      <rect x="40" y="-10" width="20" height="20" fill="#e3f2fd" stroke="#d62728"/>
      <line x1="42" y1="-8" x2="58" y2="8" stroke="#d62728" stroke-width="2"/>
      <line x1="58" y1="-8" x2="42" y2="8" stroke="#d62728" stroke-width="2"/>
      
      <rect x="40" y="20" width="20" height="20" fill="#e3f2fd" stroke="#d62728"/>
       <line x1="42" y1="22" x2="58" y2="38" stroke="#d62728" stroke-width="2"/>
      <line x1="58" y1="22" x2="42" y2="38" stroke="#d62728" stroke-width="2"/>

      <g transform="rotate(-20, 20, 20)">
        <circle cx="0" cy="10" r="8" fill="none" stroke="#d62728" stroke-width="3"/>
        <circle cx="0" cy="30" r="8" fill="none" stroke="#d62728" stroke-width="3"/>
        <path d="M7 12 L60 30 L 60 36 L 7 28 Z" fill="#d62728"/> <path d="M7 28 L60 10 L 60 4 L 7 12 Z" fill="#d62728"/> <circle cx="15" cy="20" r="3" fill="#fff" stroke="#d62728"/> </g>
    </g>
  </g>

  <line x1="755" y1="640" x2="845" y2="640" class="flow-line" stroke="#2ca02c" marker-end="url(#arrow-green)"/>

  <g transform="translate(850, 590)">
    <rect x="0" y="0" width="250" height="100" fill="url(#cnn-grad)" stroke="#1f77b4" rx="8"/>
    <text x="125" y="30" class="title-text">Compressed Model</text>
    <g transform="translate(20, 50)">
        <rect x="0" y="0" width="30" height="35" fill="#90caf9" stroke="#1f77b4"/> <text x="15" y="22" font-size="9" text-anchor="middle">Conv1'</text>
        <rect x="40" y="5" width="25" height="25" fill="#81d4fa" stroke="#1f77b4"/> <text x="52.5" y="22" font-size="9" text-anchor="middle">Pool1'</text>
        <rect x="75" y="0" width="30" height="35" fill="#90caf9" stroke="#1f77b4"/> <text x="90" y="22" font-size="9" text-anchor="middle">Conv2'</text>
        <rect x="120" y="5" width="15" height="35" fill="#b39ddb" stroke="#5e35b1"/> <text x="127.5" y="22" font-size="9" text-anchor="middle" transform="rotate(-90, 127.5, 22)">FC1'</text>
    </g>
    
    <g transform="translate(260, 50)">
      <path d="M 0 10 C 30 -20, 60 20, 10 40" fill="none" stroke="#1f77b4" stroke-width="4" marker-end="url(#arrow-blue)"/>
      <text x="70" y="25" class="label-text" font-weight="bold">Fine-Tuning</text>
    </g>
  </g>

</svg>