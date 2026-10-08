const runButton = document.getElementById("run-button");
const result = document.getElementById("result");
const invoices = [...document.querySelectorAll(".invoice")];

function markPlan() {
  invoices.forEach(card => {
    card.classList.toggle("selected", ["A","B","D"].includes(card.dataset.id));
  });
}

runButton.addEventListener("click", () => {
  runButton.classList.add("loading");
  runButton.disabled = true;
  result.classList.remove("show");
  invoices.forEach(card => card.classList.remove("selected"));

  setTimeout(() => {
    markPlan();
    result.classList.add("show");
    runButton.classList.remove("loading");
    runButton.disabled = false;
    runButton.querySelector("span").textContent = "Optimised — A + B + D";
    result.scrollIntoView({behavior:"smooth",block:"center"});
  }, 1100);
});

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add("visible");
  });
},{threshold:.08});

document.querySelectorAll(".invoice,.evidence-card,.road,.formula-card,.scale-chart,.why-card").forEach(el => {
  observer.observe(el);
});
