class MatrixRain {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.columns = [];
        this.fontSize = 20;
        this.animationId = null;
        this.isActive = false;

        this.chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*()=<>;";
        this.keywords = [
            "SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE", "DELETE",
            "CREATE", "TABLE", "INDEX", "VIEW", "DROP", "ALTER", "COUNT",
            "SUM", "AVG", "MIN", "MAX", "AND", "OR", "NOT", "NULL"
        ];
    }

    init() {
        this.canvas = document.createElement("canvas");
        this.canvas.id = "matrix-canvas";
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: black;
            z-index: 1;
            pointer-events: none;
        `;

        this.ctx = this.canvas.getContext("2d");
        this.resize();
        this.initColumns();

        window.addEventListener("resize", () => this.resize());
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.initColumns();
    }

    initColumns() {
        const columnCount = Math.floor(this.canvas.width / (this.fontSize * 1.5));
        this.columns = [];
        for (let i = 0; i < columnCount; i++) {
            this.columns.push({
                x: i * this.fontSize * 1.5,
                y: Math.random() * -50,
                speed: 0.3 + Math.random() * 0.3, 
            });
        }
    }

    getRandomChar() {
        if (Math.random() < 0.05) {
            return this.keywords[Math.floor(Math.random() * this.keywords.length)];
        }
        return this.chars.charAt(Math.floor(Math.random() * this.chars.length));
    }

    draw() {
        this.ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.font = `${this.fontSize}px monospace`;

        for (let col of this.columns) {
            const text = this.getRandomChar();
            const x = col.x;
            const y = col.y * this.fontSize;

            // color coding
            if (this.keywords.includes(text)) {
                this.ctx.fillStyle = "#ffcc00"; // SQL keywords → yellow
            } else {
                this.ctx.fillStyle = "#00ff00"; // normal chars → green
            }


            if (Math.random() < 0.1) {
                this.ctx.fillStyle = "#ccffcc";
            }

            this.ctx.fillText(text, x, y);
            col.y += col.speed;

            if (y > this.canvas.height) {
                col.y = Math.random() * -20;
                col.speed = 0.3 + Math.random() * 0.3;
            }
        }
    }

    start() {
        if (this.isActive) return;

        this.isActive = true;
        const matrixContainer = document.getElementById("matrix-container");
        if (matrixContainer && !document.getElementById("matrix-canvas")) {
            matrixContainer.appendChild(this.canvas);
        }

        const animate = () => {
            if (this.isActive) {
                this.draw();
                this.animationId = requestAnimationFrame(animate);
            }
        };
        animate();
    }

    stop() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        const canvas = document.getElementById("matrix-canvas");
        if (canvas) {
            canvas.remove();
        }
    }
}

window.matrixRain = new MatrixRain();

document.addEventListener("DOMContentLoaded", function () {
    window.matrixRain.init();

    const observer = new MutationObserver(function () {
        const matrixContainer = document.getElementById("matrix-container");
        const loginButton = document.getElementById("login-button");

        if (matrixContainer && loginButton) {
            window.matrixRain.start();
        } else {
            window.matrixRain.stop();
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
    });
});