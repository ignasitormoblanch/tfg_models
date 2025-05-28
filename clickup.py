import time
import uuid

class Task:
    def __init__(self, name):
        self.id = uuid.uuid4()
        self.name = name
        self.total_time = 0  # total time in seconds
        self.running = False
        self.start_time = None

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
            print(f"Started task '{self.name}'")
        else:
            print(f"Task '{self.name}' is already running.")

    def stop(self):
        if self.running:
            elapsed = time.time() - self.start_time
            self.total_time += elapsed
            self.running = False
            print(f"Stopped task '{self.name}'. Time added: {elapsed:.2f} seconds")
        else:
            print(f"Task '{self.name}' is not running.")

    def __str__(self):
        total_minutes = self.total_time // 60
        total_seconds = self.total_time % 60
        return f"Task '{self.name}': {int(total_minutes)}m {int(total_seconds)}s"

class TaskManager:
    def __init__(self):
        self.tasks = {}

    def add_task(self, name):
        task = Task(name)
        self.tasks[task.id] = task
        print(f"Added task '{name}' with ID {task.id}")
        return task.id

    def start_task(self, task_id):
        task = self.tasks.get(task_id)
        if task:
            task.start()
        else:
            print("Task ID not found.")

    def stop_task(self, task_id):
        task = self.tasks.get(task_id)
        if task:
            task.stop()
        else:
            print("Task ID not found.")

    def show_tasks(self):
        for task in self.tasks.values():
            print(task)

# Demo interactivo
if __name__ == "__main__":
    tm = TaskManager()
    print("Bienvenido al gestor de tareas con control de tiempo.")
    while True:
        print("\nOpciones: add / start / stop / show / quit")
        cmd = input("¿Qué deseas hacer? ").strip().lower()

        if cmd == "add":
            name = input("Nombre de la tarea: ")
            tm.add_task(name)
        elif cmd == "start":
            tid = input("ID de la tarea: ")
            try:
                tm.start_task(uuid.UUID(tid))
            except ValueError:
                print("ID inválido")
        elif cmd == "stop":
            tid = input("ID de la tarea: ")
            try:
                tm.stop_task(uuid.UUID(tid))
            except ValueError:
                print("ID inválido")
        elif cmd == "show":
            tm.show_tasks()
        elif cmd == "quit":
            print("Saliendo del gestor. ¡Hasta pronto!")
            break
        else:
            print("Comando no reconocido.")
